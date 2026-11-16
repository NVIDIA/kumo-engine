# Nightly coverage audit

The source audit below predates CI execution and withdrawal of the decimal fix.
See the [collective CI review](../PLAN.md#collective-ci-review) for current
package, lifecycle, model, profiler and dataset-access results. Historical
passing checks below do not supersede the intentional decimal failures.

Source audit: 2026-11-16, engine base `08577147` plus this branch's AML prototype.
The supplied testing plan remains the nightly specification. Current execution
scope is AML only; synthetic/MBD and the Polars performance comparison remain
deferred by the user. Weekly/on-demand expansion is outside this task.

Existing package/conformance coverage below was source-audited. The sweep rows
also record execution evidence: this increment passed 18 source/fit-anchor
cases, zero failed/skipped. Earlier runs passed 352 AML checks, 12 physical
wide-source cases, 36 package graph sweeps and 24 feature/encoding cases;
unchanged AML checks were not rerun in this increment.
All five bounded HI-small paths passed before these test increments and were not
rerun. At this audit checkpoint, the complete package/conformance/retail nightly
had not yet run. A declared case or expected refusal is not evidence that requested
behavior succeeds.

## PDF reconciliation

Reconciled against all three pages of **kumo-pytorch priorities and testing
plan.pdf**, headed *Testing strategy*, on 2026-11-16. Source SHA-256:
`e6ee1a780028594b1ff1bbfa7f382a84b844df18cd65775c5bc26370a2953655`.
This is a document/code/evidence review, not a new runtime or CI run.

Status meanings: **validated prototype** means the specified bounded cases ran
successfully; **reuse** means existing tests retain ownership (source-audited,
not a newly completed full-suite run); **partial** means some required evidence
is missing; **deferred** preserves a requirement postponed by the user.

### User decisions applied to the PDF

- All document requirements belong to the intended nightly. The PDF's final
  PR/nightly/on-demand distribution does not create a weekly or on-demand suite.
  Retain the PR trigger, as requested.
- AML is the active dataset scope. Synthetic/MBD, including synthetic
  `ltv_positive_6m`, are deferred. This does not mark them completed.
- GraphSAGE/combined use v2 anchors with the corrected cutoff; sequence-only
  paths use v1. This supersedes the starter paragraph's generic v1 reference.
- Arena remains standalone. Its S3 bucket is an authorized data source, but
  the tests have no Arena Python/check-out dependency. Test assets/models are
  outside the installed package; generic additions stay with existing tests.
- Timing/performance work and the Polars comparison remain deferred. Existing
  in-memory package tests are not a completed Polars throughput/memory comparison.

### Section-by-section result

| PDF section | Reconciled status | What is present | What remains |
| --- | --- | --- | --- |
| §1, p1: independent harness and datasets | Partial | Engine-owned test suite; [inventory](../datasets/inputs.json)/downloader for all six AML variants; verified HI-small sources, anchors and bindings | Acquire/verify and execute remaining AML variants. Synthetic/MBD deferred. Polars benchmark and matched resource/cache comparison deferred |
| §2, p1: starter workloads | Validated bounded AML prototype | All three AML tasks/five combinations: involvement with GraphSAGE/nuFormer/combined, masked values with PRAGMA, next-token with nuFormer; source identities, splits and input configurations recorded | Repeat on larger AML variants and extend beyond bounded selections. Synthetic workload deferred; extra anchor columns such as next amount/recipient are not additional starter tasks |
| §3, p2: coverage sweeps | Selected cases validated, with explicit gaps | Histories, batching, graph degree/fan-out, mixed features/encodings, empty/wide cases and independent source/anchor sizes; generic semantics retained in their existing owners, detailed below | Edge attributes are absent from the current graph declaration. Larger-than-RAM/index-memory evidence and large-scale hot-key cases remain missing. Full existing package/conformance execution still unverified |
| §4, pp2–3: benchmark stages and metrics | Partial foundations; measurement work deferred | Verified byte/file inventories, acquisition timings, overall case times, retail laps and existing index ingest timing; some runtime/seed/config metadata | The per-stage/per-workload measurement contract below, repeated cold/warm runs and uniform provenance/reporting. No timing/memory regression baseline yet |
| §5, p3: correctness | Validated bounded AML plus existing owners | Independent values/IDs/order/masks/targets/cutoff/loss checks, explicit float tolerance, controlled BPE golden and package engine checks; the empty-history consumer defect now has passing regressions | Execute the complete integrated owners and repeat correctness on expanded AML datasets. Polars performance comparison remains distinct/deferred |
| §5, p3: lifecycle | Validated bounded CPU chain; generic artifact rules reused | Train-only fitting, fresh-process reload/inference without refit/masking, frozen-input finetuning, optimizer/RNG resume; IDs/state preserved. Missing/unknown and malformed/incompatible artifact rules retain existing tests | Expanded-data/accelerator evidence and full-suite execution. Current resume verifies the next update after finetuning; arbitrary mid-epoch/native-pretraining resume is a scope limit, not a separate literal PDF requirement |
| §5, p3: two learning checks | Validated CPU prototype | Combined 50 updates and PRAGMA 100 updates, independent loss/learning gates, gradients/updates, fixed seeds/order, source/environment records and retained reports | Throughput on procured datasets is deferred. These tiny controlled runs are not generalization or dataset-throughput results |
| §5, p3: GPU diagnostics | Not implemented/validated here; deferred measurement work | No GPU claim | Fixed model/precision/hardware, prebuilt-batch versus full-training comparison, transfer/wait times, device memory and utilization interval/steady-state coverage |
| §6, p3: integrated execution and acceptance | Wired, not accepted end to end | Existing schedule/PR workflow includes package regressions, retail lifecycle, conformance and AML fixture/model checks; controlled learning reports retained | Verified CI run, scheduled downloaded-AML execution/access/storage, fixed benchmark hardware, distinct correctness/memory/timing results, variation-based thresholds and an overall required-coverage gate |

The PDF explicitly asks for selected combined configurations, not an exhaustive
cross-product. Missing combinations alone are not a reason to call the selected
sweeps incomplete. Conversely, passing selected fixtures does not satisfy the
explicit larger-than-RAM or GPU requirements.

Full-population/complete-epoch execution is an evidence limit of the current
prototype and relevant to realistic benchmarking; the PDF does not separately
require every correctness check to run a complete epoch. Likewise, a user sign-off
on every task policy is not an extra acceptance step prescribed by this document.

### Measurement requirements still open

| PDF requirement | Reconciliation |
| --- | --- |
| Ingestion elapsed/peak host memory; rows/bytes/index size; reopen separately | Some byte identities and ingest timing exist. A consistent per-workload record separating source conversion, new index construction and reopen, including peak memory, is missing |
| Materialization elapsed/peak host memory; anchors, input/target fitting, task setup; fresh/reused state | Correctness/reuse are tested; separated measurements and fresh-versus-reused comparisons are missing |
| Batch production throughput/latency; native sampling/retrieval/aggregates and Python planning/post-processing | Stage instrumentation is missing. Preserve explicit unavailable counters; do not assign unexplained wall time to Python or double-count overlaps |
| Samples/s; tokens/s; nodes/s and edges/s; actual event/output counts | No complete rate report. Tokens must count nonpadding input positions including markers/BPE pieces, excluding targets/parallel field IDs; numerical columns are not tokens |
| First-batch/p50/p95 latency; steps/s; batch wait/transfer; host/device memory | Not implemented as a common measured workload result |
| First/steady-state batches, first/later epochs, cold/warm caches; repeated runs | Not implemented as the benchmark protocol |
| Code/dependency pins, hardware, batch/output sizes, memory scope, examined/decoded rows and probe settings | Controlled learning reports and source manifests provide part of this. A uniform complete record across benchmark workloads remains missing |
| Thresholds based on observed variation; separate correctness/memory/timing failures | No calibrated benchmark thresholds or complete classification of those outcomes yet |

### Evidence and practical next steps

Recorded successful runs: **352 AML checks**, plus separate package sweeps of
**36 graph + 24 feature/encoding + 12 wide-source + 18 source/anchor cases**.
These are distinct prior commands, not a single 442-check nightly result. Five
bounded downloaded HI-small cases passed earlier, before subsequent sweep work;
they are not fresh full-population evidence. No tests or benchmarks were rerun
for this reconciliation.

The next correctness/integration work should be: run the retained owners
together on this branch, verify required tests cannot silently vanish, resolve
the edge-attribute requirement against the API, and choose the next AML expansion.
The package sweep files run when `package-regressions` runs, but package-test-only
edits are not currently in this nightly workflow's PR path filter; the AML-suite
and workflow paths are. That is a trigger-scope decision to retain explicitly,
not evidence that every change automatically launches the nightly.

At this audit checkpoint, downloaded-AML CI access/storage and a suitable
execution environment were prerequisites for scheduling its real-data path. Performance, larger-than-RAM
resource runs, GPU diagnostics and Polars stay open for the later phase; none
were removed from the PDF requirement list by the completed correctness sweeps.

## Existing execution owners

The [nightly workflow][workflow] already combines these owners:

| Owner | Runs | Reuse policy |
| --- | --- | --- |
| `package-regressions` | Existing package tests with `--require-imports`; excludes conformance and retail lifecycle, whose jobs own those checks | Keep generic regressions in their current files |
| `conformance` | Goldens, generated properties, regression corpus and recorded known gaps on both existing engines; required seeds plus 12 extra seed batches | Extend existing guarantees/generators when generic coverage is missing |
| Retail `run` / `checks` | Existing size-M lifecycle/training flows and their checks | Retain their coverage without copying fixtures or thresholds |
| AML `checks` step | Source/task/model checks, lifecycle and two controlled learning checks | Add only AML binding/model integration and the explicitly missing sweep sizes |

The AML checks and retained learning reports are wired to the existing nightly
and PR triggers. At this audit checkpoint, downloaded HI-small execution was a local CLI
prototype and CI access/execution were unverified. The linked CI review
records subsequent real-data execution with the Engine role.

## Coverage sweeps

“Reuse” below means source-level coverage exists and is selected by the workflow.
It does not mean every PDF dimension or production scale has been validated.

| Requirement | Existing coverage to reuse | Remaining work / owner |
| --- | --- | --- |
| Graph: 1–3 hops, small/large fan-outs, uniform/skewed degrees | [Sampling registry][registry] and [graph cases][graph-cases] retain generated graph coverage. The [new sweeps](aml-prototype.md#graph-coverage-sweeps) reuse those oracles for 36 cases at fan-outs 2/32/128, 1–3 hops and uniform/skewed root degrees on both engines, plus 12 AML GraphSAGE/combined cases. [Graph output tests][graph-output] retain feature/identity/orientation ownership | Implemented for these selected CPU input configurations. AML retains its two-layer binding; capped graph checks preserve complete combined histories/counts. No large-graph neural performance or uniform random sampling support is claimed |
| Graph node/edge features | [Graph representation][graph-api] exposes `node_fields`; AML encodes transaction amounts, currencies, format and time as node attributes. Existing tests verify edge connectivity | Edge attributes are not exposed by this declaration. Keep this requirement unresolved; edge indices and transaction-node features do not establish edge-attribute support |
| Histories: 16/128/1,024 events, caps/windows, one/multiple tables | Existing [caps][history-caps], [ordering][event-order] and [sequence conformance][sequence-cases] retain generic ownership. The [AML sweeps](aml-prototype.md#history-coverage-sweeps) now pass 60 cases at all three sizes, including explicit caps/windows, merged physical tables, exact source/target checks and consumer input validation | Implemented for these selected CPU input configurations. Long-sequence neural performance and full-data execution remain separate; published full-history workloads are unchanged |
| Features: count/distinct, sum, min/max, mean, variance/std, recency/mode, groups and derived values | [Reduction tests][reductions] and [registry][registry] retain basic arithmetic/grouping ownership; [materialized statistics][feature-stats] covers fit-state reuse. The [width sweep](aml-prototype.md#feature-and-encoding-coverage-sweeps) reuses the reduction fixture for 8/32/128 mixed output columns on both engines, including learned Z-score and explicit buckets. AML keeps its twelve counts/timing/calendar fields | Selected output-width cases implemented. This does not establish wide physical source rows or additional AML model feature schemas; retain those as separate boundaries |
| Encodings: vocabulary, Z-score, buckets, BPE, shared/separate state, cardinality/text length | [Input design][input-design], [field outputs][field-outputs], [text fitting][text-fitting] and [persistence][persistence] retain generic ownership. The [new sweeps](aml-prototype.md#feature-and-encoding-coverage-sweeps) add 16/256/4,096 categories per field with shared/separate state and 16/128/1,024-token texts using the existing [BPE golden][bpe-golden] | Selected CPU cardinality/text-length cases implemented. BPE training vocabulary-size scaling and large-fit memory/performance are not claimed. AML has no text input and its declared schemas remain unchanged |
| Targets/layouts: stored/generated labels, next-token, masking rates/granularities, packed/padded | [Target layouts][target-layouts], [masking][masking] and [batch API][batch-api] cover these mechanics; masking includes statistical rate checks and 0/1 boundaries. AML owns three task definitions, source-derived targets and loss eligibility | Keep generic rules in place. Current copied AML consumers use specific layouts; alternative layouts require explicit consumer compatibility checks, not automatic claims from package support |
| Scale: independently vary source rows and fit anchors, RAM-resident and larger-than-RAM indexes | [Generated sizes][generated-sizes] retain XS/S coverage. The [new matrix](aml-prototype.md#independent-source-and-fitting-anchor-sweeps) crosses 64/1,024/16,384 event rows with 8/64/512 fitting anchors on both engines: 18 bounded correctness cases verify exact source answers, train-only statistics, multiplicity and split delivery | Bounded independent-size matrix implemented. Actual RAM/index measurements, larger-than-RAM execution, large hot-key behavior and full AML population coverage remain pending. Scanning a large source does not establish a large index or out-of-core execution |
| Edge cases: hot entities, wide rows, nulls, empty histories, unseen values, duplicates, boundaries | [History caps][history-caps], [field outputs][field-outputs], [time boundaries][time-boundaries], [keyless tables][keyless] retain generic ownership. Graph sweeps include a selected 384-event hot root. The [edge-case checks](aml-prototype.md#empty-histories-and-wide-source-rows) add 20 all-empty/mixed AML cases and 12 dense/null-heavy physical 32/128/512-column cases | Selected CPU input/model checks implemented, including finite loss/backward and zero-weight pretraining behavior. The copied nuFormer empty-history mismatch is fixed with ordinary regressions. Large-scale hot-key/memory evidence remains part of source/anchor scaling |
| Batching: 32/128/512 samples, token budgets, selected combined configurations | [Native planning][length-planning] retains generic planner/layout ownership; [padding identities][padding] covers distributed graph padding. The [AML sweeps](aml-prototype.md#batching-coverage-sweeps) add 23 cases: all five combinations at the requested sample sizes, partial tails, plus eight token-budget cases including combined inputs | Implemented for selected CPU input configurations with six/seven-event histories and materialized anchors. Exact delivery, source/task alignment, budget costs and model input validation are checked; large-batch neural performance and the history × batch-size cross-product are not claimed |
| New capabilities: PK-less tables, subsecond/non-second clocks, explicit special IDs | [PK-less tests][keyless], [clock precision][clock-precision] and [special IDs][special-ids] already cover these capabilities, including native/lifecycle paths | Reuse them directly. The published AML graph binding still requires whole-second T−1s execution, and copied token consumers have explicit ID conventions. Do not change those bindings merely to exercise generic capabilities |

## Limits that must remain explicit

- [Conformance expectations][expectations] include intentional refusals, and
  [`test_known_gap`][property-tests] passes by reproducing a recorded failure.
  Neither outcome should be reported as successful feature execution.
- The [registry][registry] records graph ties at a fan-out limit whose selected
  membership can depend on index layout/build settings, including
  `GAP-GRAPH-TIES-DG`. Tests must use the documented membership semantics and
  independent cutoff rules; identical ordering/membership across builds is not
  a valid blanket requirement. Do not hide unrelated wrong neighbors or leakage.
- The same registry records unsupported combinations such as token budgets with
  `world_size > 1`, repeated physical relations within one graph layer, and timed
  intermediate traversal. Distinguish an explicit limitation from a missing
  test, and from a newly discovered defect.
- The [conformance guard][conformance-guard] fails missing, skipped, filtered or
  xfailed required obligations under `--conformance-complete`. The package
  [`--require-imports` guard][package-guard] only rejects missing-import skips;
  it is not a universal coverage-completeness check. Hardware-dependent tests
  can still skip on CPU. Required GPU coverage therefore remains unfulfilled.

## Requirements beyond the sweep matrix

| Area | Present | Still outstanding |
| --- | --- | --- |
| AML data/workloads | Six-variant acquisition inventory; HI-small files verified locally; three tasks/five model combinations run on bounded selections | Remaining AML variant execution, complete-population delivery/evaluation, final task-policy acceptance; next-amount/recipient columns are not tested tasks |
| Lifecycle | Existing generic artifact corruption/BPE/state tests and retail lifecycle; AML persistence, inference, frozen-input finetuning and fresh-process optimizer resume | Full-population and accelerator evidence; AML resume currently checks the next probe update after finetuning, not native pretraining-objective or arbitrary mid-epoch recovery |
| Training | Combined 50-step and PRAGMA 100-step fixed-batch learning checks, with source/loss references and retained reports | Full-data learning/throughput is separate; fixed masks and repeated examples do not establish generalization |
| Stage measurements | Acquisition timings, total AML/case elapsed times, retail stage laps; [index manifests][materialization] already record `ingest_seconds` | Per-stage peak host memory, reopen/fresh-vs-reuse separation, samples/tokens/graph rates, latency distributions, native/Python breakdowns and cold/warm repetitions. Reuse existing counters where available; unavailable is not zero |
| GPU diagnostics | Some existing hardware-gated package checks | Fixed device/precision/model comparison of prebuilt-batch model execution versus full training, memory and utilization sampling/coverage; no GPU run claimed |
| Nightly acceptance | Existing owners, AML PR trigger and tiny-fixture execution, retained learning reports | Verified CI execution, downloaded AML job/data access, storage/hardware selection, explicit required-coverage reporting and distinct correctness/memory/timing outcomes |
| Polars comparison | Deferred by the user | Return during performance work using equivalent workload semantics; do not require identical batches. Existing package/conformance engine tests retain their current ownership |

## Recommended implementation order

1. **History-length coverage implemented:** 60 cases pass at 16/128/1,024 events
   for PRAGMA, nuFormer and combined input paths, reusing independent task helpers.
   Full-history cases and explicit capped/windowed variants are separate.
2. **Batching coverage implemented:** 23 cases cover 32/128/512 samples, partial
   tails and selected token budgets, including combined input alignment. Generic
   batch planning and padding algorithms stay owned by existing tests.
3. **Graph degree/fan-out coverage implemented:** 36 package and 12 AML cases
   pass; edge attributes remain unresolved against the current API.
4. **Feature/encoding sweeps implemented:** 24 package cases cover selected
   output widths, category counts and text lengths, retaining existing owners.
5. **Selected edge cases implemented:** empty/mixed AML histories and physical
   wide-row projection checks, retaining generic null/duplicate/cutoff owners.
6. **Bounded source/fit-anchor matrix implemented:** 18 cases cover independently
   selected sizes. Larger-than-RAM/resource evidence remains incomplete.
7. **PDF reconciled:** see the [section-by-section result](#pdf-reconciliation).
   Next correctness/integration work is retained-owner execution, required-coverage
   checks, the edge-attribute decision and AML expansion. Measurement/resource
   work remains deferred; verify CI data access before scheduling downloads.

The original audit added no tests. Subsequent increments should state their exact
cases, reuse existing oracles, and record actual outcomes before marking a matrix
requirement complete. History, batching and AML graph checks use the existing
nightly model-check job, together with empty-history checks; generic graph,
feature/encoding, wide-source and source/anchor sweeps use
`package-regressions`.

[workflow]: ../../../.github/workflows/kumo-pytorch-t2b.yml
[registry]: ../../../packages/experimental/kumo-pytorch/conformance/registry.py
[graph-cases]: ../../../packages/experimental/kumo-pytorch/conformance/graph_cases.py
[graph-output]: ../../../packages/experimental/kumo-pytorch/tests/test_diskgraph_graph_output.py
[graph-api]: ../../../packages/experimental/kumo-pytorch/src/kumo/pytorch/contract/batch/representations.py
[history-caps]: ../../../packages/experimental/kumo-pytorch/tests/test_history_cap_pushdown.py
[event-order]: ../../../packages/experimental/kumo-pytorch/tests/test_event_output_order.py
[sequence-cases]: ../../../packages/experimental/kumo-pytorch/conformance/sequence_cases.py
[reductions]: ../../../packages/experimental/kumo-pytorch/tests/test_graph_time_and_reductions.py
[feature-stats]: ../../../packages/experimental/kumo-pytorch/tests/test_materialized_feature_stats.py
[input-design]: ../../../packages/experimental/kumo-pytorch/tests/test_input_design.py
[field-outputs]: ../../../packages/experimental/kumo-pytorch/tests/test_field_outputs.py
[text-fitting]: ../../../packages/experimental/kumo-pytorch/tests/test_text_from_index.py
[persistence]: ../../../packages/experimental/kumo-pytorch/tests/test_fitted_state_persistence.py
[bpe-golden]: ../../../packages/experimental/kumo-pytorch/conformance/spec/guarantees/G-ENCODING-001/golden/C-ENCODING-003.json
[target-layouts]: ../../../packages/experimental/kumo-pytorch/tests/test_target_layouts.py
[masking]: ../../../packages/experimental/kumo-pytorch/tests/test_masking.py
[batch-api]: ../../../packages/experimental/kumo-pytorch/tests/test_batch_api.py
[generated-sizes]: ../../../packages/experimental/kumo-pytorch/conformance/generate.py
[time-boundaries]: ../../../packages/experimental/kumo-pytorch/tests/test_time_boundaries.py
[keyless]: ../../../packages/experimental/kumo-pytorch/tests/test_keyless_tables.py
[length-planning]: ../../../packages/experimental/kumo-pytorch/tests/test_native_length_planning.py
[padding]: ../../../packages/experimental/kumo-pytorch/tests/test_padding_identity_native.py
[clock-precision]: ../../../packages/experimental/kumo-pytorch/tests/test_clock_precision.py
[special-ids]: ../../../packages/experimental/kumo-pytorch/tests/test_special_token_ids.py
[expectations]: ../../../packages/experimental/kumo-pytorch/conformance/expectations.json
[property-tests]: ../../../packages/experimental/kumo-pytorch/tests/conformance/test_properties.py
[conformance-guard]: ../../../packages/experimental/kumo-pytorch/tests/conformance/conftest.py
[package-guard]: ../../../packages/experimental/kumo-pytorch/tests/conftest.py
[materialization]: ../../../packages/experimental/kumo-pytorch/src/kumo/pytorch/materialization.py
