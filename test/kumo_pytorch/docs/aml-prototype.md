# AML HI Small task and check guide

HI Small has five bindings: E01 GraphSAGE, E02 Combined and E05 nuFormer for
binary involvement, E03 PRAGMA for masked-value pretraining, and E04 nuFormer
for next-token pretraining. They share the [catalog](../workloads/bindings.json),
native AnchorTable path and test-owned models. Full population, training,
source correctness, histories and lifecycle have distinct execution purposes.

## Scope and outputs

Fetch/verify the eleven pinned inputs through [the data commands](../README.md#data).
Run campaigns through the commands in [the suite guide](../README.md).
Bounded checks preserve complete histories for selected roots; they do not
claim complete-population acceptance. The population campaign iterates every
eligible anchor without a limit. Reports retain source/anchor identities,
configuration, fitted state, stage timings, assertions and child diagnostics.

## Identity, clocks and fitting

Accounts use stable identities from the pinned binding table. GraphSAGE and
Combined use v2 anchors and the graph execution cutoff one second before the
original prediction time; other HI Small bindings use v1. Assertions retain the
original prediction time and enforce strict past-only inputs. Labels and split
metadata never enter model inputs.

Fit only on the declared training split with an explicit encoder cutoff. Stored
columns and aggregate encoders have distinct fitting populations. Validation,
test and inference reuse fitted inputs; fingerprints and no-refitting guards
check that state stays fixed. Production batches come from
`task.batches(split=..., epoch=..., output_format="model")`, without requests.

## Account-involvement checks

[involvement.py](../involvement.py) calculates anchor labels and input expectations
from original transaction rows. It checks sender/receiver histories, twelve
windowed counts, graph eligibility/features, fitting statistics and original
prediction clocks. Self-transfers and duplicate physical rows keep their declared
multiplicity. Graph comparisons follow identities and connectivity.

[Task fixtures](../model_checks/test_involvement.py) independently verify binary
cross-entropy, eligible/padded examples and analytical gradients. Source-derived
labels are checked before conversion. Loss and evaluation use the same eligible
population; exact structure/count checks are separate from floating tolerance.
No unspecified classification threshold or model-quality metric is invented.

## PRAGMA masked-value checks

[masked_values.py](../masked_values.py) reconstructs original packed IDs from raw
source fields and saved training encoders, without production encoding or mask
helpers. It checks history membership, calendar/age values, field ownership,
mask eligibility, corrupted inputs and original supervised values.

The HI Small policy uses token/event/key selection probabilities 0.15/0.10/0.10,
combined by union, 0.10 unknown replacement and one supervised position per
nonempty sample. Null fields are omitted; supervised UNK and reserved-ID remapping
remain explicit. [Fixtures](../model_checks/test_masked_values.py) use independent
scalar CE with 0.1 label smoothing and analytical gradients. Padding, ignored
labels and unsupervised positions contribute no loss or gradient.

## Next-token checks

[next_token.py](../next_token.py) and [reference_tokens.py](../reference_tokens.py)
reconstruct the stream independently from source values and saved encoders.
For `[a,b,c]`, inputs are `[a,b]` and targets `[b,c]`; there is no second shift or
fabricated terminal successor. Markers and original unknown IDs remain eligible.
Padding uses PAD inputs, ignored targets and false masks.

[Fixtures](../model_checks/test_next_token.py) check exact alignment, full histories,
cutoffs, masks, independent CE/gradients and zero-target behavior. Causality checks
change later tokens and require earlier predictions/other samples to stay fixed;
a deliberately bidirectional model must fail. Targets cannot influence inputs.

## Save/load and inference

[lifecycle.py](../lifecycle.py) persists fitted task state and model weights,
reopens the same index without ingestion in a fresh process, and checks identical
batches and predictions under the same CPU configuration. Targets are absent,
fitting is forbidden and held-out categories remain unknown. PRAGMA inference
removes masking; next-token inference restores complete histories.

## Finetuning and resume

[training_lifecycle.py](../training_lifecycle.py) checks frozen-input finetuning
in a fresh process. HI Small pretraining models replace their head with binary
involvement and use the separate forecast training anchors; supervised models
retain their head. Input fingerprints, content and fitting provenance stay fixed.
A subsequent fresh process restores model/AdamW/RNG state and must reproduce the
uninterrupted next update. Wrong optimizer state and refitting must fail.

## Mid-epoch training resume

[checkpoint_resume.py](../checkpoint_resume.py) retains the original objective,
including pretraining masks. It saves `TaskBatches.state_dict()`, weights,
optimizer and RNG at an interior position, then compares the entire remaining
suffix with fresh-process restoration. Check inputs, targets, masks, IDs/order,
loss, model/optimizer state and final iterator state exactly. No seeking-speed
claim is implied. [Fixtures](../model_checks/test_checkpoint_resume.py) cover
multiple positions, shuffling, chunk boundaries, short tails and corrupted state.

## Controlled learning checks

[Learning fixtures](../model_checks/test_learning.py) hold one native batch fixed:
Combined trains for 50 steps and PRAGMA for 100. They require finite computation,
branch/head updates, unchanged fitted inputs and improvements beyond uniform and
majority baselines. These overfitting tests complement the [real-data 100-update
acceptance](training-acceptance.md); they do not establish unseen-data quality.

## History coverage sweeps

[History fixtures](../model_checks/test_history_sweeps.py) cover 16/128/1,024 events,
explicit caps/windows and merged physical tables. [Source probes](../history_checks.py)
exercise sender/receiver windows and filters with independently calculated answers.
Equal-time events may permute, but chronology, membership, multiplicity, cutoff and
clipping eligibility remain required.

## Batching coverage sweeps

[Batch fixtures](../model_checks/test_batching_sweeps.py) cover 32/128/512 samples,
partial tails and selected token budgets, including combined inputs. Existing
native planning/padding tests retain generic ownership. Late unsupported-length
errors remain failures, not accepted preflight refusals.

## Graph coverage sweeps

[Graph fixtures](../model_checks/test_graph_sweeps.py) and package sampling tests
cover fan-outs 2/32/128, one-to-three hops and degree skew. Source expectations
validate nodes, features and connectivity independent of tensor ordering and
eligible tied-neighbor choices. Combined histories/counts stay complete even
when graph fan-out is capped. Large-graph runtime is a separate benchmark claim.

## Feature and encoding coverage sweeps

Existing package tests cover 8/32/128 output columns, 16/256/4,096 categorical
values, shared/separate encoder state and 16/128/1,024-token text. The controlled
BPE golden case remains independent of AML's categorical/numerical features.
Expanded bindings exercise their own declared additions; see the
[feature map](binding-coverage.md).

## Empty histories and wide source rows

[Empty-history cases](../model_checks/test_empty_histories.py) cover all-empty and
mixed batches, finite supervised loss and differentiable zero-weight pretraining.
Physical source fixtures vary 32/128/512 columns and dense/null-heavy values.
Unseen values, duplicates and clock boundaries remain explicit traps. Intentional
zero-row ingestion rejection follows the [approved refusal policy](required-coverage.md#capability-refusals).

## Independent source and fitting-anchor sweeps

The package matrix independently varies 64/1,024/16,384 event rows and 8/64/512
fitting anchors, checking source answers, train-only statistics and delivery.
The [25k/400k experiment](fitting-memory.md) measures fitting memory separately.
Neither bounded matrix proves larger-than-RAM execution.

## Validation and current limits

Read results for the exact tested revision. Fixture success does not certify
real-data, complete-population, CUDA or performance acceptance. Source checks,
training, lifecycle and profiling can fail independently; preserve each result
and its diagnostics. The [coverage map](coverage-audit.md) and
[roadmap](../PLAN.md#remaining-acceptance-work) track outstanding acceptance.
