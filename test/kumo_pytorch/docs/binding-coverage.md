# Binding coverage expansion

The user approved adding LI Medium, Synthetic Medium and MBD incrementally in
PR #353 on 2026-11-17, superseding the earlier synthetic-data deferral. Retain all
existing tests, including supervised HI Small nuFormer, which the new PDF omits
from its four-row baseline. Weekly testing remains deferred.

Source: *kumo-engine-arena-binding-coverage-plan.pdf*, five pages, 2026-11-17.
Its Arena revision is
[`220aa187`](https://github.com/kumo-ai/kumo-engine-arena/blob/220aa187921ddb1d846985bec8f8edff1d3581c9/kumo_specs/workload-selection.v1.json).
Its Engine audit revision `9432cddf` is historical provenance; execution must
test the PR's actual revision. [bindings.json](../workloads/bindings.json) pins
the 16 PDF bindings plus E05, the existing supervised HI Small nuFormer case.
It retains selectors, representations, split columns, target meanings and source
catalog hashes. E05 and extended contracts such as S01/S04/M04 have suite identities, not new Arena
registrations. Extended contracts retain their original `base_binding_id` as
provenance. Registration is not execution coverage.

## Base bindings

GraphSAGE/combined use v2 anchors; PRAGMA/nuFormer use v1. Dataset versions are
independent: AML v8, Synthetic/MBD v1. Original input variants are `default`, or `trx` for MBD; S01 uses the suite BPE extension and S04 the richer/grouped-aggregate extension.

| Ref | Dataset / population | Workload | Model | Status / next work |
| --- | --- | --- | --- | --- |
| E01 | HI Small / account-daily | account-involvement-next24h | GraphSAGE | Existing tests; retain outstanding CI failures |
| E02 / a001 | HI Small / account-daily | account-involvement-next24h | Combined (Fusion) | Existing tests; retain outstanding CI failures |
| E03 / a017 | HI Small / account-daily | masked-value-pretrain | PRAGMA | Existing tests; retain outstanding CI failures |
| E04 / a018 | HI Small / account-daily | next-token-pretrain | nuFormer | Existing tests; retain outstanding CI failures |
| E05 | HI Small / account-daily | account-involvement-next24h | nuFormer | Existing additional supervised case retained |
| L01 | LI Medium / account-daily | next-outgoing-amount-currency | Combined | Required feature, target, training and delivery checks added; CI acceptance pending |
| L02 | LI Medium / account-daily | next-outgoing-delay | nuFormer | Fixture coverage retained; real-data campaigns disabled by user (L01 only) |
| L03 | LI Medium / account-daily | next-outgoing-format | GraphSAGE | Fixture coverage retained; real-data campaigns disabled by user (L01 only) |
| L04 | LI Medium / account-daily | account-involvement-next24h | PRAGMA | Unmasked supervised fixture coverage retained; real-data campaigns disabled by user (L01 only) |
| S01 | Synthetic / weekly-account | account-masked-value-pretrain | PRAGMA | One BPE contract with required source checks and shared campaigns; real-data CI pending |
| S02 | Synthetic / weekly-customer | customer-next-token-pretrain | nuFormer | Native customer-history correctness, training, delivery and lifecycle registered; real-data CI pending |
| S03 | Synthetic / weekly-account | aml_graph_user_last30d | GraphSAGE | Port event-membership graph and retrospective binary target |
| S04 | Synthetic / ltv_positive_6m | ltv_positive_6m | Combined | Required positive-LTV task with richer/grouped aggregates and four shared campaigns; real-data CI pending |
| M01 | MBD / product-uptake | uptake-joint | Combined | Port four independently valid binary targets |
| M02 | MBD / transaction-forecast | next-transaction-amount-currency | PRAGMA | Port mixed targets and independent validity |
| M03 | MBD / transaction-forecast | next-transaction-category | nuFormer | Port event type/subtype targets including auxiliary target checks |
| M04 | MBD / transaction-forecast | future-trx-count | GraphSAGE | Stored and generated count checks in four shared campaigns; real-data CI pending |

## Feature additions within each selected binding

Use one binding per row in *dataset-model-feature-coverage 3.pdf*. Extend that
binding's contract and add independent checks together; do not schedule separate
base and extended bindings. Retain the Arena base ID as provenance and use a
suite identity for the extended contract. Reuse existing sources and anchors.
Previously enabled coverage remains in place.

| Base | Additional declarations and assertions |
| --- | --- |
| S01 | BPE on an existing string field such as dispute reason; shared compatible encoders across event families; capacity for extra pieces. Adds F09, broadens F10. |
| S04 | Amount reductions, distinct counts, mode, recency, grouped/nested counts by mcc/channel and ratios. Adds F06, broadens F05/F07. Raw outputs are checked; numerical outputs enter the shared Combined model. |
| M03 | Composite tokens, packed output, aligned labels from withheld fields, threshold labels and matching-group IDs. Withhold target fields from every exposing input. Adds F16, broadens F10/F11. Loader checks first. |
| M04 | Separately named Kumo-generated future count, checked against stored labels on valid anchors using the exact registered window. Adds F15. Loader checks first. |

F01 identity, F02 point-in-time history and F08 scalar encodings apply throughout.
F03 sequences, F04 graphs, F05 aggregates, F07 derived values, F10 tokens and F11
layouts apply where declared. L01 aggregate coverage means counts only; S04 adds reductions and grouping;
existing token/layout tests do not certify every option. F12 masked prediction,
F13 next-token prediction and F14 supervised targets keep separate assertions.
F17 candidates remains a documented limitation with refusal checks, not successful
execution coverage. F18 lifecycle/delivery reuses shared checks on selected bindings.

## L01 feature coverage

The user selected AML LI Medium / Combined from *dataset-model-feature-coverage
3.pdf* (pages 1, 3 and 6). Complete each binding's feature requirements before
starting the next binding. Previously added bindings remain in place.

L01 uses shared `nightly run` campaigns for bounded correctness (`smoke`), training
and full-population delivery. Correctness and training require these checks on
every selected anchor in train, valid and test:

| Named check in reports | Features | Independent expected answer |
| --- | --- | --- |
| `identity_cutoff` | F01, F02 | Original account identities, graph clocks and input allowlist |
| `graph_membership` | F02, F04 | Raw historical transactions, sender/receiver edges, counterparties and sample ownership |
| `graph_values` | F04, F08, F11 | Raw account/transaction rows encoded using saved state; scalar values, validity, types and shapes |
| `history_membership` | F02, F03, F10, F11 | Complete historical event groups, separators, valid prefixes and source positions |
| `history_values` | F03, F07, F08, F10, F11 | Whole encoded event records, direction, age and UTC calendar values; chronology with equal-time permutations allowed |
| `aggregate_counts` | F05 | All twelve historical counts and train-only normalization, calculated from raw transactions |
| `aggregate_layout` | F05, F11 | Declared scalar float32 values, boolean validity and shapes |
| `stored_targets` | F14 | Original anchor labels, exact float32 projection, independent masks and training-only category vocabulary |
| `forecast_targets` (smoke only) | F14 | Earliest outgoing raw transaction in the forecast window, including its exact decimal amount |

The correctness campaign additionally checks both labels against the earliest
outgoing source transaction in `[T,T+24h)`, breaking ties by physical source row.
Both labels must come from that one transaction. Exact source decimals are checked
before the separate float32 transport check. Transport fixtures and training
also exercise independently missing targets; their supplied labels need not be
generated from their tiny input histories. Input assertions run again on training
probes, optimizer batches and checkpoint evaluation. Existing strict two-repeat,
100-update acceptance and independent loss/evaluation denominator checks remain
required outside this feature table.

The feature factory reuses the existing account/transaction source and token
oracles. Dataset identity comes from the catalog. Other source formats can
declare independent feature oracles through the same runner hook, without adding
dataset runners or model wrappers. Full-population delivery retains streaming
checks and does not claim this bounded source-feature comparison. JSON and
Markdown reports include each assertion's status and checked-anchor counts by
split and phase (population, training/probes, evaluation). JSON also records its
feature IDs, meaning and independent reference. Missing, failed or partial
assertion evidence fails the summary, even if the broad feature counters pass.
Every assertion runs on a batch before its collected failures are raised.

These real-source probes cover this binding's full historical graph neighborhood;
finite-fanout sampling variants remain covered by fixtures. Encoding checks apply
saved state independently to raw values; they do not certify every input fitting
algorithm. The bounded real-source checks supplement the existing fixture sweeps.

The user confirmed on 2026-11-18 that equal-time history order may vary. The
package's type/text primary-key comparison explains the initially observed order;
the former numeric-order assertion was too restrictive for this suite. Required
cases now accept both permutations while rejecting changes to event membership,
multiplicity or individual values, and any reordering across timestamps. The
forecast-target selector retains its separately specified source-row tie rule.
Real LI Medium correctness and training acceptance still require CI execution.

Local validation (2026-11-18, compatible native extension): all 396 focused
checks passed, including 26 L01 feature cases and its two-repeat, 100-update
training check. All 432 required CPU cases collect exactly; five GPU-owned cases
remain delegated. Formatting, import ordering, Flake8, repository documentation
and license checks passed. Independent review found no remaining blockers in this
increment. These figures describe the preceding fixture increment, not real-data acceptance.

## Earlier expansion sequence

1. **LI contracts and anchors:** reuse the native projection and input builders;
   add the four target definitions, per-output masks, train-only vocabularies and
   strict v2 clocks. This increment adds required native fixture checks across
   two epochs. It does not claim real-data, length-cache or timing acceptance.
2. **LI model and real-data integration:** add numerical/categorical heads and
   unmasked PRAGMA supervision, real-data selection, CLI/report integration,
   smoke/full populations, lifecycle/training and timing campaigns. Exercise
   native AnchorTable planning/cache reuse wherever that coverage is claimed.
3. **Synthetic:** add each selected row with its required feature additions
   and independent boundary fixtures: S02, S01 with BPE, then S04 with grouped
   aggregates.
4. **MBD:** pin the required `trx` sources and anchors; add M04 with generated
   counts and M03 with auxiliary target checks, one row at a time.
5. **Complete accounting:** map required assertions to execution owners, reject
   missing tests and retain real-data, full-population and profiling evidence.

Code, models and assets remain test-owned, with no Arena runtime dependency.
DiskGraph answers use independent expectations; Polars/GPU references remain
timing-only. Accepted refusals, successful execution and defects remain distinct.
This expansion does not authorize package runtime changes.

## Data and access

LI Medium already has 11 pinned objects in [inputs.json](../datasets/inputs.json):
611,594,783 source bytes and 2,385,945,191 bytes including anchors and mappings.
Reuse the verified downloader and its acquisition timing.

Arena's pinned dataset catalog lists 20,738,597,153 source bytes for Synthetic
Medium and 77,598,007,519 for all MBD modalities. These are catalog sizes, not
measured transfers or final workload footprints. Select MBD's required `trx`
files before scheduling downloads; do not automatically fetch every modality.

The original [CI grant](../datasets/ACCESS.md) covers HI Small's 11 objects. New
datasets require a reviewed read-only grant for their exact selected objects.
The [LI Medium addition](../datasets/ci-li-medium-read-policy.json) grants only
the ten additional pinned objects; the dataset manifest is already covered. It
was applied with user authorization on 2026-11-17 and verified by rereading the
bucket policy; both existing statements were preserved. No local real-data
download is part of validation. Effective CI-role downloads still need a run.

## LI heads and real-data jobs

`nightly run` owns both datasets through the same catalog and runner. It reuses
the verified downloader, declared source mapping, Parquet AnchorTable projection
and existing model backbones. All
iteration uses native `task.batches(split=..., epoch=..., output_format="model")`
without explicit requests. Models and runners remain test-owned.

Training selects the first 32 anchors per forecast split without inspecting
labels, preserving each selected root's complete source history. Each of two
same-seed CPU repetitions performs 100 eligible updates, checks finite loss and
gradients, every trainable tensor's update and decreasing probe loss, then
requires exact replay. Validation/test scores use the saved trained checkpoint
and frozen training-label vocabularies. Missing labels exclude only their output;
unseen non-null categories remain eligible UNKNOWN IDs. Zero-eligible batches
never update parameters.

L01 sums independently normalized Smooth L1 amount and categorical cross-entropy
losses. L02 uses Smooth L1 for raw delay seconds, L03 categorical cross-entropy,
and L04 binary cross-entropy without masked-value pretraining. Smooth L1 uses
beta=1 in original numerical units and float64 loss arithmetic, preserving small
improvements on large float32 targets; there is no fitted target scaling. Every
training/evaluation batch rechecks source targets, not only the first epoch.

Full mode has no anchor or source limit. It checks every selected forecast-split
anchor's identity/order, clocks, target values and masks, with a partial report
written every 100 batches. It is a delivery campaign, not full-dataset training.
Full-population acceptance is recorded only after all three splits finish.

The categorized binding matrices use the candidate DiskGraph wheel for 38 jobs:
HI Small's five bindings × training/smoke/full, and LI Medium's four bindings ×
training/full, plus L01 bounded correctness, lifecycle and source histories, and S01/S02/S04 correctness, training, full delivery and lifecycle. Each publishes JSON/Markdown reports;
training adds loss curves and checkpoints. The summary requires all 38 reports and verifies
their revision, binding identity and completion.
Declared early refusals are labelled separately from executed campaigns.

HI Small's five bindings additionally retain fitting-memory checks (three
repetitions each), but no longer enable model profiling, index-reuse timing or
DiskGraph/Polars/GPU loader timings. L01 retains its four benchmark campaigns.
The combined summary therefore requires 47 reports: 38 dataset campaigns,
four L01 benchmarks and five combined HI Small fitting-memory reports.

Local tiny-fixture validation does not establish real LI acceptance or runtime
on its 52,332,325-row anchor population. CI-role access, full-source resource use and
Linux execution still need CI validation. L01 now requires lifecycle/iterator restoration, history probes and DiskGraph
profiling. Execution gaps remain failures, as described below. Synthetic/MBD remain the following dataset increments.

## Shared structure

HI Small and LI Medium are separate dataset records, with independent inventories,
source paths, anchors and enabled campaigns. No runner branches on either name.
The same structure accepts unrelated table names, root IDs, anchor columns and
prediction clocks. Tiny shop fixtures exercise all four model families through
both the shared runner and benchmark factory.

| Responsibility | Existing shared location | New dataset supplies |
| --- | --- | --- |
| Selection and planning | `bindings.py`, `workloads/bindings.json` | Dataset, task, input/anchor profile, model family and campaigns |
| Acquisition and sources | `data.py` | Pinned inventory, table keys/clocks/relations, identity mapping and optional raw-format converter |
| Native anchors | `population_anchors.py` | Identity/split/time/target columns and execution cutoff |
| Inputs and objectives | `contracts.py`, `targets.py` | History/graph/aggregate declarations and target types/validity |
| Models | `models/` | Existing backbone plus binary, categorical, regression or pretraining objective |
| Execution | `nightly.py`, `population.py`, `training_acceptance.py` | Binding and campaign selection |
| Profiling | `benchmarking/kumo.py`, `benchmarking/` | The same binding; backend and measurement settings |
| Reporting | `nightly.py` | Required campaign declarations; missing or mismatched evidence fails |

Already normalized Parquet uses the shared preparation directly. Source-specific
anchor-manifest checks are explicitly declared; arbitrary datasets need only
the shared pinned-file inventory. History capacities sum declared families and
respect each relation's join column; graph fanouts use their own observed degrees.
Other history paths use conservative source-row bounds, which may overallocate
until tighter bounds are declared. Native objectives require compatible models
and whole-second execution boundaries; invalid declarations fail before execution. Raw-format
conversion remains schema-specific: `aml_source.py` converts the published bank
transaction format for either independently configured dataset. It does not own
campaigns, anchors, targets or models. Future formats can provide a converter
without adding dataset-specific runner or profiler adapters.

Existing focused HI fixtures and lifecycle/history CLIs retain their expected
answers and domain scenarios. They are not templates to copy for another dataset.
L01 enables the shared lifecycle helpers and benchmark adapter; other LI
bindings retain fixture coverage but their real-data campaigns are now disabled. Other Synthetic/MBD entries remain
pending; complex validity rules and contract variants still need implementation.


## L01 campaign expansion

Only AML LI Medium / Combined is expanded in this increment. Its catalog enables:

- Detailed training profiling with stage timers, counters, memory and GPU metrics.
- Index construction followed by two reopens, and fresh versus reused fitted state.
- DiskGraph loader timing; no LI Polars or GPU reference comparison.
- A separate native length-planning and second-epoch cache probe. The ordinary
  combined benchmark preserves native graph/history iteration with fixed batches.
- Fresh-process inference, frozen-input finetuning with the original amount/currency
  task, optimizer continuation and exact mid-epoch iterator restoration.
- Independent real-source multi-history windows, limits and filters. Equal-time
  events may permute, including at a limit; membership, multiplicity and timestamp
  ordering remain required. Declared preflight filter refusals are reported explicitly.
- Existing fixture sweeps: 16/128/1024-event histories, caps/windows/merged tables,
  uniform/skewed graph degrees and 2/32/128 fanouts, sample/token budgets, and
  empty/mixed histories. Targets and head schemas remain those of L01.

Every enabled campaign is required in the combined summary. HI's existing
grouped scenario (`legacy-lifecycle` in the catalog) has separate Lifecycle and
Correctness matrix entries. The source-history sweep runs in Correctness,
independently of lifecycle; its commands, bounded source checks and report
artifacts are retained. The catalog runner reuses its persistence helpers. All additions are test-owned.

Local probes expose two package gaps: combined mid-epoch reload changes the
materialized task fingerprint, and length-based combined batching falls back
from the native typed/history iterator. Both checks require successful execution
and remain failures. The separate planning job prevents that gap from suppressing
normal detailed profiling and index/state-reuse measurements. Real-data CPU/GPU
validation remains a CI responsibility. Fitting scalability, other LI bindings,
Synthetic and MBD are unchanged by this increment.


Validation (2026-11-18, local compatible native extension): the 477-case CPU run
produced 472 passes and five failures, all exercising the two package gaps above
(three existing HI resume failures and two L01 checks). All 133 fixture sweeps
passed. The final feature/graph/training rerun passes all 48 cases, including the
new negative-edge-index regression and actual training-report evidence validation.
All 275 harness checks pass. The inventory now collects exactly 478 CPU cases;
five GPU cases remain delegated to CI. Independent review has no remaining
findings. Live LI data/GPU execution awaits publication and a subsequent CI run.

## Harness semantics and diagnostics

The follow-up stays in test code and reuses shared assertions:

| Area | Acceptance rule |
| --- | --- |
| LI numerical targets | Verify original Decimal labels against source transactions before conversion, then compare the projected values exactly at float32 precision. |
| Floating aggregates | Only declared float64 means and moments allow the existing 1e-12 absolute/relative tolerance. Counts, masks and structure stay exact. |
| Keyless ties | Check eligible values, membership and limits separately for each prepared dataset. |
| Graph serving | Match source identities, remap edge indices, check features and require a complete legal layer-by-layer sampling witness, including tied fanout alternatives. Shared by generated graphs and H&M. |
| Native anchors | The former explicit-request equivalence case now materializes a Parquet AnchorTable and verifies two native epochs against original source rows. |
| Sequence ties | Independent source answers enforce chronology, cutoff, values, multiplicity and both component/merged limits. Equal-time permutations and eligible boundary alternatives may differ. |
| Failure evidence | Report the first mismatched anchor/target, batch and source position, expected/actual values and comparison precision. |

No package runtime behavior changes. The required dataset/model inventory now
contains 482 CPU cases and five delegated GPU cases. Local validation covers
59 target/precision/population cases (one accepted preflight refusal), 39 sequence
obligations across 13 seeds, 60 semantic/fault-injection checks, and the complete
128-case serving/anchor regression run (four accepted preflight refusals). The
serving run completed in 41.49 seconds after moving the independent aggregate
oracle outside the per-anchor loop. Declared refusals verify rejection, not
execution of an unsupported capability. Documentation links, required-inventory
collection, scoped lint/format checks and repository license checks also pass.
The independent reviewer found no remaining issues; an exhaustive independent
check agreed with the graph validator on 7,680 outcomes from 120 tiny cases.
These results do not replace a real-data Linux/GPU nightly run.

## S02 / Synthetic Medium / nuFormer

The first Synthetic row in *dataset-model-feature-coverage 3.pdf* uses the registered
`customer-next-token-pretrain` binding, weekly-customer v1 anchors and dataset v1.
Its binding ID ends in `ca3566e`. S01/PRAGMA and S04/Combined are described below. S03 is not selected by this PDF.

The existing catalog, downloader, native anchor projector, nuFormer model,
training checks, lifecycle checks and categorized workflow matrices are reused.
`relational_sources.py` handles declared direct-owner, owned-entity and two-endpoint
public Parquet events; `history_features.py` independently reads original rows.
Neither component branches on a dataset ID, binding ID or model name. Future
bindings supply source/ownership and input declarations rather than new runners.
No package-runtime source changed. The shared worker accepts the catalog runner
and free-space floor; it still installs CPU Torch for correctness/training.
S02 uses the existing RTX PRO 6000 group with a provisional 128 GiB free-space
floor after setup and before download. The acquisition artifact includes measured
capacity and free bytes; insufficient space fails before any dataset transfer.
This floor is not a measured peak-storage guarantee. Direct PR runner admission
remains separately deferred; real-data storage and runtime need CI validation.

The pinned inventory contains 1,613 objects: 1,610 public Parquet shards across
customers, accounts and seven event families, the published source manifest,
and the weekly-customer anchor manifest/payload. Public source bytes total
17,606,082,087; the anchor payload adds 107,928,518 bytes. Truth tables, mutable
risk/closure fields and unrelated public tables are excluded. The download
verifies exact byte counts and SHA-256 values. The user-approved GetObject grant
covers only these nine versioned public-table prefixes and the three exact
metadata/anchor objects, with no list/write permission.

Bounded campaigns retain complete selected-customer histories across all owned
accounts. Full delivery selects every anchor and all source roots. Transfers
with two accounts owned by the same customer appear once, with `self` direction;
other endpoints receive their own incoming/outgoing event. Source mapping retains
cutoff/future events for DiskGraph to reject at sampling time. Root identity
is the original nonnegative integer, not a newly assigned row number.

| Required assertion | Feature IDs | Independent answer |
| --- | --- | --- |
| `identity_cutoff` | F01, F02 | Original customer identity, sample identity and unchanged prediction clock |
| `history_values` | F01, F02, F03, F07, F08, F10, F11 | Original rows joined by ownership; strictly earlier complete histories, source values, UTC age/calendar, saved scalar encodings, separators, source positions and tensor validity |
| `next_token_targets` | F11, F13 | Original complete token stream shifted once, including its final target; exact padding/eligibility |

Equal-time whole-event records may permute; cross-time order, multiplicity and
values stay strict. Heterogeneous events contribute only their declared fields.
`Tokens(missing="error")` rejects NULL values in declared fields; it does not
require fields belonging exclusively to another event type. Corruption tests
exercise both allowed ties and rejected ordering, ownership, values, masks,
terminal targets and source-index changes.

Correctness/training require individual evidence for every selected train,
validation and test anchor. Training retains two reproducible 100-update CPU
runs and source checks on optimizer/probe/evaluation batches. Lifecycle reuses
native next-token loss/target validation with frozen input fitting. No Synthetic
benchmark campaigns are enabled in this feature increment. Real-data execution,
resource use and full-population acceptance still require CI evidence.

Source preparation preserves schema for empty event families; any DiskGraph
capability refusal remains the package’s responsibility. Two-endpoint event
identities require source IDs in `[0, 2**62 - 1]`; larger values fail preparation
with the family/column and supported bound before arithmetic overflow.


## S01 / Synthetic Medium / PRAGMA

One selected binding extends Arena's `account-masked-value-pretrain` contract.
`suite:S01:bpe-v1` keeps the original ID ending `abbca33` as `base_binding_id`.
Weekly-account v1 anchors, public sources, masking policy and PRAGMA consumer are
reused. The contract now declares `Text` BPE for `disputes.reason`, with vocabulary
size 256, minimum frequency 2 and empty-string behavior `empty`. Packed row tokens
have no scalar-field token cap, so every BPE piece reaches the model. Compatible
scalar fields share one fitted encoder across event families; field IDs still
identify the source family and field.

The same catalog, source mapper, native AnchorTable projector, training runner,
lifecycle runner and workflow matrices serve S01 and S02. A declared account
view changes the root from customer to account. Financial events stay with their
account; owner-level events are expanded to each selected account. A transfer
between sibling accounts is outgoing/incoming, while an actual same-account
transfer is `self` once. IDs stay stable when selecting a bounded root population.
Strictly earlier histories exclude equal/future events; stored-column encoder
fitting separately follows the package's inclusive table cutoff, across entities.
The shared projector sets that fitting cutoff one second before the latest
selected training prediction clock.

| Required assertion | Feature IDs | Independent answer |
| --- | --- | --- |
| `identity_cutoff` | F01, F02 | Original account/sample identity and prediction clock |
| `history_values` | F01, F02, F03, F07, F08, F10, F11 | Original rows joined by ownership; complete eligible events, scalar encodings, field identities, positions, packed offsets and padded event layout; UTC calendar values and age |
| `bpe_pieces` | F09, F10 | Apply saved BPE merge ranks and ID mapping independently to original text; require all pieces, positions and field identities without truncation |
| `masked_targets` | F11, F12 | Original pre-corruption IDs, masked/unknown positions, reconstruction labels and eligible loss positions |

Equal-time whole-event records may permute only within their eligible timestamp
group. Bijection checks preserve multiplicity even when masking hides a value.
The BPE oracle uses the saved tokenizer only for preprocessing; merge selection
and ID lookup are independent of its encoder. Fixtures include hand-calculated
merges and corruption cases for pieces, values, masks, ownership, chronology,
offsets, positions and derived outputs. Shared 100-update training checks retain
finite losses/gradients, every trainable parameter update and loss decrease.

The inventory shares 1,611 pinned public/source-manifest objects with S02 and
adds the weekly-account anchor manifest and 141,305,955-byte anchor payload.
The payload declares 58,264,001 anchors. The separately authorized GetObject
statement covers only these two additional objects; the existing grant is
preserved. No new workflow, dataset/model wrapper, Arena dependency or package
runtime change is introduced. S01 uses the same RTX runner and 128 GiB storage
preflight with CPU training. No Synthetic timing campaign is enabled.

Correctness, training, full-population and lifecycle campaigns are required.
Real-data CI execution, storage/runtime measurements and full-population
acceptance remain pending; local fixtures do not establish those results.

## S04 / Synthetic Medium / Combined

One binding, `suite:S04:aggregates-v1`, extends the registered positive six-month
LTV task (base ID ending `8925fa`). It shares S02's customer source view and uses
the pinned `ltv_positive_6m/v2` anchors. Prediction is 28 days after onboarding;
graph execution uses `T-1s` so whole-second events at `T` remain excluded.
History ages and recency outputs are restored to the original prediction clock.
The original anchor label must equal `gross_profit > 0` before float32 transport.
This checks the stored threshold, not the financial simulation generating profit.

The shared contract/model builders now accept multiple graph node types. S04
samples all seven historical event families from the customer root, with no
counterparty expansion. It combines their graph features with the complete token
history, 35 base counts (seven families across five windows) and numerical
aggregate additions. The extension declares amount sum/mean/min/max/population
spread, distinct merchant-category/channel counts, mode, recency, grouped counts,
nested channel counts by merchant category and a recent-history ratio. Mode and
intermediate values are preparation checks; the other numerical outputs enter
the existing Combined aggregate encoder. No extra model wrapper is introduced.

| Required assertion | Features | Independent answer |
| --- | --- | --- |
| `identity_cutoff` | F01, F02 | Original root/sample identities, onboarding prediction time and v2 execution clock |
| `graph_values` | F01, F02, F04, F08, F11 | Original source identities, ownership, historical membership, encoded features and connectivity, independent of tensor order |
| `history_values` | F01, F02, F03, F07, F08, F10, F11 | Complete original event records, age/calendar, scalar tokens, separators, positions and allowed equal-time permutations |
| `aggregate_values` | F05, F06, F07, F08, F11 | Python reductions/groups over original rows; exact raw counts and masks, numerical tolerances and train-only normalization |
| `stored_targets` | F14 | Original profit threshold and projected label, eligibility and binary target encoding |

`aggregate_features.py` supplies independent reduction answers. `graph_features.py`
reuses the package's keyed-graph assertion with original source rows; it imports
no reference engine. The existing history oracle handles both next-token and
supervised sequences. Hand-calculated fixtures cover unequal groups, each window
boundary, zero denominators, empty recent histories, cutoff/future poison values,
allowed tensor permutations and rejected corruption. Numerical comparisons keep
integer counts exact; float64 reductions allow 2e-12 relative/absolute rounding,
and float32 normalization uses the existing 2e-6 bound. Raw output dtypes come
from declarations, independent of fitted state. Mode checks verify train-only
vocabulary fitting and encoded outputs; unseen categories map to UNKNOWN, which
does not distinguish their original strings. A native three-way tied-mode fixture
checks the declared value-order rule with a known category.

The four campaigns use the same workers, RTX storage preflight and CPU training
as S01/S02. The inventory shares their 1,611 public/source-manifest objects and
adds only the LTV v2 manifest and 10,903,428-byte anchor payload (206,776 anchors).
The approved two-object GetObject grant was applied and reread successfully.
No workflow or package-runtime change is required; no Synthetic timing campaign
is enabled. Real-data execution and storage measurements remain for CI.

Local native fixtures execute smoke, full delivery and two reproducible 100-update
training runs. Fresh-process finetuning and reload/inference run, but mid-epoch
iterator resume reports the existing package error, `this state was saved for a
different task`. That lifecycle test remains required and failing. The harness
also now hashes empty/singleton broadcast tensors correctly and uses the existing
scalar binary loss oracle in fresh-process finetuning.

Validation on the compatible local native extension: 37 of 38 S04 cases pass;
the required lifecycle case fails as above. The broader native regression has
194 passes and three Combined resume failures (S04, HI Small and LI Medium).
All 347 harness checks pass, and all 591 required CPU cases collect exactly
(five GPU cases remain delegated). Formatting, license and documentation checks
pass. Two independent review rounds leave no remaining findings after adding
dtype-corruption regressions and native mode-tie coverage. These are local
fixture results, not real-data CI acceptance.


## M04 / MBD / GraphSAGE

One binding, `suite:M04:generated-count-v1`, extends the registered `trx`
GraphSAGE forecast (base ID ending `fa6e68e`). Training still predicts the stored
`future_count` using the existing numerical head and Smooth L1 loss in original
count units. A separate `generated_count` target asks Kumo to count transactions
in the same 30-day future window. It is checked but never included in the loss
or exposed as a model input. No new model wrapper or workflow is added.

Original monthly prediction time is T; graph execution is E=T-1s. Transaction
clocks retain original nanoseconds for source assertions and floor to seconds
for native execution. Since prediction/window boundaries are whole seconds,
native `(E,E+30 days]` is equivalent to original `[T,T+30 days)`. Historical
inputs remain strictly before T. Duplicate transactions count separately. A
covered empty future window has a valid count of zero; coverage means the
recorded global span covers the window, not proof that collection has no gaps.

| Required assertion | Features | Independent answer |
| --- | --- | --- |
| `identity_cutoff` | F01, F02 | Original client membership and fold split, prediction and execution clocks |
| `graph_values` | F01, F02, F04, F08 | Original transaction IDs, complete eligible neighborhood, features, masks and connectivity independent of tensor order |
| `stored_targets` | F14 | Exact original integer count equals original future rows; separate float32 transport check |
| `generated_targets` | F15 | Native int64 count and validity equal an independent count of original nanosecond events |

The shared converter for partitioned string-root events uses the original five
client-membership files and 1,000 transaction files from folds 0–4. It excludes
fold -1 and unrelated modalities. Client IDs are UTF-8-sorted ranks; event IDs
are stable file/physical-row offsets independent of bounded/full selection.
This intentionally avoids Arena's global chronological renumbering, which is
not needed for the complete graph neighborhood. Identities are internal; no
exact Arena task fingerprint or tensor-order parity is claimed. Numeric category
codes become canonical strings; nonfinite values become missing. Fold, balance
flags and labels are not graph features. Root tables are explicitly non-temporal.

`partitioned_sources.py` handles this source format; `forecast_features.py`
independently reads original rows and checks graph/count forecasts through the
existing feature hook. Native AnchorTable projection, model, campaign, lifecycle
and reporting components are reused. The original integer count is retained
alongside float32 training transport, so large counts are not compared after
rounding. Full-population delivery also checks generated versus exact stored
counts; the more expensive original-row feature oracle remains bounded to the
correctness/training campaigns, as for other bindings.

The 1,008 pinned objects total 7,028,272,242 bytes: 6,037,620,860 source bytes,
plus the extraction manifest, v2 anchor manifest and 980,925,125-byte anchor
payload. Sources contain one million clients and 385,304,108 transactions;
anchors contain 11 million examples. The approved GetObject grant covers five
exact membership keys, five transaction partition prefixes, and three exact
manifest/anchor keys, with no listing or writes. It was applied and reread.
The shared RTX worker uses CPU training and an approved 128 GiB free-space
preflight. Actual index growth and full real-data runtime remain unmeasured.

Local native fixtures pass correctness, complete delivery and both 100-update
training repetitions. Reload/inference and fresh-process finetuning pass;
mid-epoch resume exposes the existing package task-identity error (`this state
was saved for a different task`). That required lifecycle test remains failing.
Real-data CI acceptance is pending. M03 remains inactive for the next increment.

Validation: `model_checks/test_count_forecast.py` has 29 passes and the one
required lifecycle failure above. All 348 harness checks pass. Regression checks
across existing bindings have 118 passes and only the previously recorded S04
and L01 lifecycle failures. All 621 required CPU cases collect; five GPU cases
remain delegated. Independent review verified all 1,005 source pins against the
original extraction inventory and found no remaining issues after adding clock
corruption and empty-selected-history regressions. No package runtime files
changed in this increment.


## Current campaign selection and lifecycle organization

AML Medium runs only L01 / LI Medium / Combined. L02, L03 and L04 remain in the
catalog with their fixture tests, but no real-data campaigns are enabled. HI
Medium has no active binding. Correctness, training, population, lifecycle and
benchmark planning all use this same selection on PR, nightly and manual runs.

HI Small has three grouped matrix entries: lifecycle in Lifecycle, and source
correctness plus history sweeps in Correctness. All five workload/model
combinations remain, with four anchors per split, two-example batches and one
thread. Correctness validates original-source labels, input membership, masks,
targets and losses without optimizer updates. Lifecycle performs one setup update
before persistence, inference, finetuning and exact resume. The existing `all`
CLI mode remains available locally, but CI explicitly selects each owner.
Each grouped job has a 45-minute timeout, its own reports and independent failure
status. There are 36 per-binding campaigns plus three grouped entries, four L01
benchmarks and 15 HI Small fitting repetitions.

### Lifecycle ownership after extraction

| Owner | Checks |
| --- | --- |
| HI Small Correctness / smoke | Original-source labels; input membership, masks, targets and loss oracles (`run_aml --campaign correctness`) |
| HI Small Correctness / histories | Independent windows, filters, ordering and limit sweeps |
| H&M Correctness | Future-row cutoffs, generator labels, sampling-tie eligibility and empty histories (`tests/e2e/test_correctness.py`) |
| H&M Training/evaluation | Baseline identity, AUROC bands/seed spread, score distribution and active/dormant ranking (`tests/e2e/test_training.py`) |
| Lifecycle | Fresh-process reload/inference, serving equivalence, frozen-input finetuning, artifact compatibility and exact resumed batches |

H&M's `tests/e2e/conftest.py` registers shared fixtures once; `hm_support.py` owns
their native helpers. Its original combined
scored-anchor/quality test is split into delivery under Lifecycle and distribution
under Training, preserving every assertion. Setup trains/persists/infers/serves
each flow/seed once. Independent checks jobs for both scales consume those
artifacts with `KUMO_PYTORCH_E2E_REUSE_ONLY=1`; missing runs fail instead of
retraining. All six check jobs and the producer matrix are required by the gate.
Reports use category-specific artifact names to prevent collisions.

The shared binding runner's four-anchor-per-split delivery check remains a
bounded integrity check, not a full-population campaign. Lifecycle launches no
standalone fitting-memory or timing benchmark; elapsed times are diagnostics.
No package runtime behavior, oracle values or tolerances changed.

Validation: 364 harness checks and 11 HI Small native runner cases pass,
including all five correctness-only probes; one Combined mid-epoch resume case
fails with `this state was saved for a different task`. The same case fails with
the pre-extraction `run_aml.py` from `98583e45`, confirming the existing package
identity issue remains exposed. All 19 lightweight H&M dataset/reuse tests pass;
both tiers collect 48 campaign cases, and all 626 required CPU cases collect
(five GPU cases delegated). Independent review verified all 45 original H&M
assertions are preserved and each shared fixture is registered once. Workflow,
Python and documentation checks pass. Full H&M training and real-data CI were
not rerun locally.
