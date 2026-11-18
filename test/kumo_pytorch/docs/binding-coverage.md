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
catalog hashes. E05 is a suite identity, not a newly registered Arena binding. Registration is not execution coverage.

## Base bindings

GraphSAGE/combined use v2 anchors; PRAGMA/nuFormer use v1. Dataset versions are
independent: AML v8, Synthetic/MBD v1. Input variants are `default`, or `trx` for MBD.

| Ref | Dataset / population | Workload | Model | Status / next work |
| --- | --- | --- | --- | --- |
| E01 | HI Small / account-daily | account-involvement-next24h | GraphSAGE | Existing tests; retain outstanding CI failures |
| E02 / a001 | HI Small / account-daily | account-involvement-next24h | Combined (Fusion) | Existing tests; retain outstanding CI failures |
| E03 / a017 | HI Small / account-daily | masked-value-pretrain | PRAGMA | Existing tests; retain outstanding CI failures |
| E04 / a018 | HI Small / account-daily | next-token-pretrain | nuFormer | Existing tests; retain outstanding CI failures |
| E05 | HI Small / account-daily | account-involvement-next24h | nuFormer | Existing additional supervised case retained |
| L01 | LI Medium / account-daily | next-outgoing-amount-currency | Combined | Required feature, target, training and delivery checks added; CI acceptance pending |
| L02 | LI Medium / account-daily | next-outgoing-delay | nuFormer | Native targets, heads and training/full-delivery jobs added; CI acceptance pending |
| L03 | LI Medium / account-daily | next-outgoing-format | GraphSAGE | Native targets, heads and training/full-delivery jobs added; CI acceptance pending |
| L04 | LI Medium / account-daily | account-involvement-next24h | PRAGMA | Unmasked supervised head and training/full-delivery jobs added; CI acceptance pending |
| S01 | Synthetic / weekly-account | account-masked-value-pretrain | PRAGMA | Port base, then BPE variant |
| S02 | Synthetic / weekly-customer | customer-next-token-pretrain | nuFormer | Port owned-account history, contract and model |
| S03 | Synthetic / weekly-account | aml_graph_user_last30d | GraphSAGE | Port event-membership graph and retrospective binary target |
| S04 | Synthetic / ltv_positive_6m | ltv_positive_6m | Combined | Port positive six-month LTV binary task, then aggregate variant |
| M01 | MBD / product-uptake | uptake-joint | Combined | Port four independently valid binary targets |
| M02 | MBD / transaction-forecast | next-transaction-amount-currency | PRAGMA | Port mixed targets and independent validity |
| M03 | MBD / transaction-forecast | next-transaction-category | nuFormer | Port event type/subtype targets, then aligned-target variant |
| M04 | MBD / transaction-forecast | future-trx-count | GraphSAGE | Port stored count, then generated-count variant |

## Distinct contract variants

Preserve base bindings and assign separate suite variant identities. Do not
regenerate sources or anchor tables or imply Arena registered these extensions.

| Base | Additional declarations and assertions |
| --- | --- |
| S01 | BPE on an existing string field such as dispute reason; shared compatible encoders across event families; capacity for extra pieces. Adds F09, broadens F10. |
| S04 | Amount reductions, distinct counts, mode, recency, grouped/nested counts by mcc/channel and ratios. Adds F06, broadens F05/F07. Loader outputs first; training needs explicit model changes. |
| M03 | Composite tokens, packed output, aligned labels from withheld fields, threshold labels and matching-group IDs. Withhold target fields from every exposing input. Adds F16, broadens F10/F11. Loader checks first. |
| M04 | Separately named Kumo-generated future count, checked against stored labels on valid anchors using the exact registered window. Adds F15. Loader checks first. |

F01 identity, F02 point-in-time history and F08 scalar encodings apply throughout.
F03 sequences, F04 graphs, F05 aggregates, F07 derived values, F10 tokens and F11
layouts apply where declared. Existing aggregate coverage means counts only;
existing token/layout tests do not certify every option. F12 masked prediction,
F13 next-token prediction and F14 supervised targets keep separate assertions.
F17 candidates remains a documented limitation with refusal checks, not successful
execution coverage. F18 lifecycle/delivery reuses shared checks on selected bindings.

## Current increment: L01 only

The user selected AML LI Medium / Combined from *dataset-model-feature-coverage
3.pdf* (pages 1, 3 and 6). Complete each binding's feature requirements before
starting the next binding. Previously added bindings remain in place.

L01 uses shared `nightly run` campaigns for bounded correctness (`smoke`), training
and full-population delivery. Correctness and training require these checks on
every selected anchor in train, valid and test:

| Features | Independent assertions |
| --- | --- |
| F01, F02, F04 | Original account identities, strict historical membership, sender/receiver edges, counterparties and graph clocks |
| F03, F07, F10 | Chronological history groups, allowing either order within a timestamp; direction, age and UTC calendar values; complete encoded event values and separators |
| F05 | All twelve historical counts, inclusive lower window bounds, self-transfers and training-only count normalization |
| F08, F11 | Graph scalar encoding against saved state; tensor dtypes, aligned masks/shapes, valid token prefixes, source event indices and scalar piece positions |
| F14 | Independent amount/currency validity, stored values, training-only target vocabulary and loss/evaluation denominators |

The correctness campaign additionally checks both labels against the earliest
outgoing source transaction in `[T,T+24h)`, breaking ties by physical source row.
Both labels must come from that one transaction. Transport fixtures and training
also exercise independently missing targets; their supplied labels need not be
generated from their tiny input histories. Input assertions run again on training
probes, optimizer batches and checkpoint evaluation. Existing strict two-repeat,
100-update acceptance remains required.

The feature factory reuses the existing account/transaction source and token
oracles. Dataset identity comes from the catalog. Other source formats can
declare independent feature oracles through the same runner hook, without adding
dataset runners or model wrappers. Full-population delivery retains streaming
checks and does not claim this bounded source-feature comparison. JSON and
Markdown reports include checked-anchor counts per feature and split; missing
or partial feature evidence fails the summary.

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
increment. No real-data CI run, commit or push was performed for this increment.

## Earlier expansion sequence

1. **LI contracts and anchors:** reuse the native projection and input builders;
   add the four target definitions, per-output masks, train-only vocabularies and
   strict v2 clocks. This increment adds required native fixture checks across
   two epochs. It does not claim real-data, length-cache or timing acceptance.
2. **LI model and real-data integration:** add numerical/categorical heads and
   unmasked PRAGMA supervision, real-data selection, CLI/report integration,
   smoke/full populations, lifecycle/training and timing campaigns. Exercise
   native AnchorTable planning/cache reuse wherever that coverage is claimed.
3. **Synthetic:** pin the necessary sources/anchors; port the four bases with
   independent boundary fixtures, then implement S01/S04 variants separately.
4. **MBD:** pin required `trx` sources and anchor populations; port the four
   bases and their validity rules, then M03/M04 variants separately.
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

One dataset-campaign matrix uses the candidate DiskGraph wheel for 24 jobs:
HI Small's five bindings × training/smoke/full, and LI Medium's four bindings ×
training/full, plus L01 bounded correctness. Each publishes JSON/Markdown reports;
training adds loss curves and checkpoints. The summary requires all 24 reports and verifies
their revision, binding identity and completion.
Declared early refusals are labelled separately from executed campaigns.

Local tiny-fixture validation does not establish real LI acceptance or runtime
on its 52,332,325-row anchor population. CI-role access, full-source resource use and
Linux execution still need CI validation. LI lifecycle/iterator restoration,
length-cache experiments and timing adapters remain subsequent work; the new
jobs do not claim them. Synthetic/MBD remain the following dataset increments.

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
LI lifecycle and timing campaigns remain disabled until their assertions and
runtime are validated. Synthetic/MBD entries remain explicitly pending; complex
validity rules and contract variants still need implementation.
