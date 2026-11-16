# Real AML prototype

Run from the engine repository root, with the checkout's `kumo.pytorch` and a
matching DiskGraph installation in the PyTorch test environment:

```sh
python test/kumo_pytorch/data.py fetch --dataset hi-small
python -m test.kumo_pytorch.run_aml \
  --output test/kumo_pytorch/.runs/aml-prototype
```

The first command needs existing read access to the Arena bucket; use
`--profile NAME` if needed. The runner itself uses local files and does not
download or publish anything. It verifies all eleven pinned inputs before use.
All implementation code is owned by this suite; no Arena checkout is needed.

## Scope and outputs

The default run selects the first four physical anchor rows in each of train,
valid and test, separately for `forecast_split` and `split`. Selection does not
filter by label or by usable history. It runs all five workload/model cases,
one training batch and the selected validation/test batches, on CPU/DiskGraph.
Every case must have usable targets, finite outputs and gradients, and an actual
optimizer update. Failures produce a nonzero exit and a failed report.
After training, each case saves its fitted task and model and verifies inference
in a fresh process on the selected validation/test anchors.

`--anchors-per-split`, `--batch-size`, `--steps`, `--threads` and `--seed` control
the bounded run. `--steps` is an upper bound on training batches available in
the selected anchors, not an epoch count. No histories are trimmed to meet it.
This is not a complete AML epoch, convergence check or performance benchmark.

- Prepared tables default to `.data/prepared/hi-small-probe/`. They can be reused
  only with identical pinned inputs and selected roots, after output checksums
  pass. Changing the selected roots requires a new `--prepared-dir`.
- `--output` must name a new directory. It holds selected original anchor rows
  in `anchors.json`, a combined `report.json`, per-case results and native indexes.
- Reports include source hashes/counts, mapping scope, original sample IDs,
  selected split sizes, target weights, mean losses, fitted-state fingerprints
  and optimizer update counts. Missing or unusable cases fail; they are not skips.
- Each case's `lifecycle/` folder retains the Kumo artifact, bound weights,
  replay configuration, expected inference tensors, child-process log and result.
  These are local run artifacts, not package or Git contents.

## Identity, clocks and fitting

[Source preparation](../aml_source.py) is copied from Arena's `kumo_specs/aml_source.py`
at revision `b0701af6075de1cf79f0064d1f8213cc9fb4fbc6`, with only its schema-helper
import redirected here. It scans every source transaction and keeps every
transaction touching the selected accounts at every source time. Counterparty
accounts supply endpoints; their unrelated histories are outside this view.

Internal account IDs are ranks in the complete checksum-pinned account binding.
Canonical string identities retain bank/account distinctions and leading zeros.
Transaction IDs retain physical source row positions. A self-transfer becomes
one history event, with both sender/receiver graph relationships preserved.
Amounts remain exact `decimal128(38,10)` until native numerical encoding;
timestamps use the original minute values interpreted as UTC. Source laundering
flags and patterns never become model input features.

[Anchor loading](../aml_inputs.py) preserves labels, null validity, sample IDs and
split membership. GraphSAGE/combined use v2 `graph_as_of_time = T - 1 second`,
while retaining original T as `prediction_time_ns` metadata. Sequence-only cases
use v1. [Shared declarations](../contracts.py) translate combined history/count
windows and ages back to the original prediction-time semantics.

Only training requests are passed to `materialize(..., fit_split="train")`.
The explicit encoder cutoff is the greatest whole second before the latest
selected training prediction. Under Kumo's API, stored-column encoders fit the
prepared source rows up to that cutoff; aggregate encoders fit the selected
training requests. Validation/test batches use that same fitted task without
refitting. A fingerprint check detects changed fitted state after evaluation.

## Account-involvement checks

The independent [reference](../involvement.py) reads raw source transactions,
without importing source mapping, contract-building or model loss helpers.
It verifies the selected v1/v2 anchors against the task rules from Arena revision
`b0701af6075de1cf79f0064d1f8213cc9fb4fbc6`:

- The label is any laundering transaction sent or received in `[T, T+24h)`.
  Self-transfers count once. A negative requires a fully observed window.
- Daily UTC anchors need prior activity. Forecast windows must stay within their
  split and global observation end. Split boundaries use the full source's
  timestamp quantiles, with boundary ties assigned to the later split.
- Model inputs use only history strictly before original T. Graph execution
  uses v2's T-minus-one-second clock. Graph nodes/edges match raw endpoints;
  sequence lengths/event groups match full history; combined counts match all
  twelve declared windows/roles and training-anchor normalization.
- Emitted labels/validity match the source reference. Independent scalar BCE
  and eligible-example counts match each model's loss, excluding padding and
  missing labels. Reports retain the loss numerator and counts separately.

The [task fixtures](../model_checks/test_involvement.py) exercise all three models
on both engines, with positives, negatives, missing targets, padding, zero-target
backward passes, excluded gradients and deliberately corrupted targets/masks/
clocks/losses/inputs. Changing future values or labels must leave inputs unchanged;
changing a historical field must change its expected model input position.
The [reference fixtures](../model_checks/test_involvement_reference.py) cover
horizon boundaries, maturity, quantiles, endpoint identity and known BCE values.

Existing package tests own generic padding identity, token encoding and lifecycle
behavior. These AML checks add task integration; they do not independently decode
every sequence token or prove all possible leakage is absent. Event provenance
indices are compared relative to each sample because row assembly and native
columnar assembly use different index origins. Padding is excluded from AML
evaluation rather than interpreted as another source anchor.

The task's classification metrics/thresholds and parts of its input/evaluation
specification remain draft. This prototype checks its current uniform BCE and
input declarations; it does not define AUROC gates or claim full task acceptance.
The real run checks 12 involvement anchors (four per split), all negative under
the unchanged first-row selection. Positive-label evidence comes from fixtures.
The other two task checks are described below. Reports distinguish their bounded
checks from full-population acceptance.

## PRAGMA masked-value checks

The [PRAGMA reference](../masked_values.py) reconstructs original packed value IDs
from source transactions and the saved train-fitted encoder/vocabulary state.
It does not call Kumo's masking/encoding helpers or the model's target selector
to compute expected answers. Generic fitting and random-mask algorithms remain
owned by the existing package tests; this checks their AML task integration.

Source history must be strictly before the original cutoff, in chronological
source-row order, with one event for a self-transfer. Checks compare the six
permitted fields, packed event lengths/owners, calendar values and ages. Null
fields are omitted, and fitted vocabulary/bucket mappings determine original
targets. No label, split or future field is an input.

The prototype declares token/event/key masking probabilities 0.15/0.10/0.10,
combined by union, with 0.10 unknown replacement and at least one supervised
position per nonempty sample. Masked positions supervise the original value;
unknown replacements provide context corruption without supervision. An original
UNK value is eligible and maps to PRAGMA's swapped UNK ID when supervised.
The corruption flag reports changed bytes, independently of supervision.

[Task fixtures](../model_checks/test_masked_values.py) pin this declaration,
exercise deterministic all-selected policies, missing and previously unseen
values, and compare selected head positions without using the production
selector. Independent scalar log-sum-exp checks CE with 0.1 uniform label
smoothing; analytical gradients check its weighting. Padding, invalid targets
and `-100` are excluded, including zero-target backward passes. Corrupted
targets, masks, fields, clocks, losses and denominators must fail. Future/evaluation
field changes leave fitted training inputs and targets unchanged.

The bounded real run validates 132 train, 229 validation and 187 test supervised
positions across four anchors per split. Reports retain independent loss sums
and eligible/unsupervised/padded token counts; the common `invalid` count includes
unselected positions, not just missing data. Absent fields have no packed token.
These are checks of the current encoded objective, not reconstruction in original
numeric units, calibrated model-quality metrics or full-population acceptance.
The upstream task/input/evaluation specifications still contain draft choices.

## Next-token checks

The [next-token reference](../next_token.py) reconstructs the complete historical
stream from raw transaction values and the saved fitted encoders. The small
[token reference helper](../reference_tokens.py) is shared with PRAGMA; it uses
no production encoding or target-generation helpers. Fitting itself remains
covered by package tests.

Each event contributes age, direction, amounts/currencies, payment format and
calendar tokens, with one separator between events. For original stream
`[a, b, c]`, Kumo must emit inputs `[a, b]` and targets `[b, c]`. This prototype
has no prefix token. The terminal original token remains the last answer; it
has no input position demanding a fabricated successor. Separators and original
unknown values remain eligible targets. Right-padding uses PAD inputs, ignored
`-100` targets and false masks. The explicit missing-field policy is an error,
not dropping a field or anchor.

[Task fixtures](../model_checks/test_next_token.py) check exact alignment, source
order, full-history lengths, single-event histories, held-out unknown values,
uniform per-position CE and analytical gradients. Invalid/ignored targets and
padded samples contribute neither loss nor gradient; all-excluded batches have
a differentiable zero loss. Corrupted shifts, targets, masks, inputs, cutoffs,
losses and denominators fail. Changing future/evaluation source fields leaves
historical inputs and targets unchanged.

Causality is checked on the copied nuFormer through the AML model interface:
changing a later token suffix must leave earlier predictions and other samples
unchanged while changing suffix predictions. Deliberately enabling bidirectional
attention must fail that check. Altering target values or IDs under padding
masks cannot affect predictions. These tests exercise the consumer, complementing
the existing package's generic layout tests.

The bounded real run verified 850 train, 2,082 validation and 1,477 test target
positions, across four anchors per split. Reports include independent CE sums
and eligible/excluded counts. The common `invalid` counter includes right-padding
slots; `padding` counts slots belonging to padded requests. No second shift or
label smoothing is applied. This does not establish a trained model-quality
threshold, GPU causality validation, finalized draft policies or full-population
acceptance.

## Save/load and inference

The [lifecycle check](../lifecycle.py) uses Kumo's existing `Artifact`,
`save_weights` and `load_weights` APIs. After the optimizer update, it saves the
fitted task and trained weights and establishes an inference baseline in the
training process. A separate Python interpreter loads the artifact and weights,
reopens the same DiskGraph index with ingestion disabled, and infers on the same
validation/test requests with `inference=True` and no supplied labels.

Fitting entry points are guarded in both inference runs: an attempted fit fails
even if it would reproduce the same vocabulary. The complete fitted-state hash
must remain unchanged, covering encoder statistics, vocabulary and reserved IDs.
Every batch must omit targets. PRAGMA must contain no MASK tokens; its user
representations and trained MLM head outputs over all unmasked fields are checked.
nuFormer receives full histories rather than the shortened training inputs.
These pretraining outputs are persistence diagnostics, not objective-loss metrics.

Input/metadata hashes and predictions match exactly before and after reload
under the same CPU, thread count and batch settings. This is a persistence check
within one engine, not a requirement that different engines produce identical
batches. The child records its process identity and selected package path, and
delivery must preserve every selected sample ID. All five real cases passed on
eight validation/test anchors each.

The existing raw-source pipeline tests now also cover this lifecycle. A category
first observed after training stays unknown after reload. Known full-history
lengths catch residual next-token shortening, and changing a correctly bound
checkpoint's weights must fail prediction equality. Generic artifact corruption,
compatibility and missing-value behavior retain their existing package tests.
Complete-population inference and GPU validation remain pending; there are no
AML text/BPE fields in this prototype.

## Finetuning and resume

The [training lifecycle check](../training_lifecycle.py) reloads the saved task
and weights. PRAGMA and next-token nuFormer retain their trained backbone and
replace the pretraining head with a binary account-involvement head. The three
supervised cases retain their existing heads. This downstream PRAGMA task is a
lifecycle check, not an additional starter-workload catalog entry.

All five paths use only the selected `forecast_split=train` anchors, retaining
v2 execution clocks for graph models. `fitted=artifact.fitted_state.inputs`
preserves vocabularies, special IDs and numeric state while allowing the label
target to bind. Input fitting is guarded, input fingerprints and fitting
provenance must remain unchanged, and only involvement labels are emitted.
PRAGMA masking is off and nuFormer consumes full, unshifted histories.
Independent involvement loss checks are reused; backbone parameters must update.

After one finetuning update, the probe saves the task, bound model weights,
AdamW state (including moments, steps and hyperparameters), CPU RNG and the next
probe epoch. The uninterrupted process performs its next update; a fresh Python
process restores the checkpoint without fitting and performs the same update.
Sample IDs, input hashes, loss, model state and optimizer state must match
exactly under the same CPU/thread/batch configuration. Resume artifacts and
child logs live under each case's ignored `finetuning/` output directory.

The existing raw-file fixture uses different pretraining and forecast training
anchors. A category first seen at finetuning stays unknown; full-history lengths
and target counts are checked. Deliberately resetting AdamW moments and steps
must fail comparison against uninterrupted training, even with correct model
weights. Generic artifact validation remains in the existing package tests.

All five real HI-small paths pass. These checks take the first batch of probe
epochs 0 and 1, with four selected forecast training anchors by default. They
verify resume after finetuning; they do not establish arbitrary mid-epoch
iterator recovery, native pretraining-objective resume, convergence or GPU
equivalence. The separate controlled learning checks are described below.

## Controlled learning checks

[Two learning checks](../model_checks/test_learning.py) reuse the tiny AML fixture
and existing model/task oracles. DiskGraph produces one batch of two training
requests. Fitting ends strictly before their original prediction time, including
the v2 clock translation for combined. The batch is then held fixed, including
PRAGMA's epoch-0 corruption; each optimizer step consumes the same inputs and
targets. No downloaded data or Arena runtime is required.

| Model/objective | Steps | Acceptance | Observed local result |
| --- | --- | --- | --- |
| Combined involvement | 50 | Both binary labels correct; loss above its zero floor reduced by at least 80% | 2/2 correct; BCE 0.69334 → 2.93e-9 |
| PRAGMA masked values | 100 | At least 90% of selected values correct; excess smoothed CE reduced by at least 80% | 16/17 correct; loss 4.92152 → 0.98540; theoretical floor 0.81268 |

The smoothed-CE floor is the entropy of the target distribution, not zero:
the answer has probability `0.9 + 0.1/V` and each other vocabulary entry has
probability `0.1/V`. Both checks must beat uniform prediction and the fixture's
majority-label accuracy. An untouched model copy must retain its initial outputs.
Existing independent BCE/CE checks verify every step's loss and eligible count.
Losses, gradients and parameters must stay finite; every named input branch and
head must receive a nonzero first-step gradient and update. The fitted state and
complete batch must remain unchanged.

Confident negative logits expose float32 cancellation in PyTorch BCE. Its
comparison to the scalar float64 reference retains relative tolerance `2e-6`
and uses absolute tolerance `max(2e-7, mean eligible-logit float32 ULP)`.
This bound follows floating-point spacing; ignored logits cannot enlarge it.
A regression verifies float64 agreement and rejection of a `1e-5` loss error.
The model's objective and package implementation are unchanged.

Runs use width 16, model seed 17, batch seed 7, AdamW at learning rate 0.01 with
zero weight decay, float32 CPU computation, one thread and deterministic Torch
algorithms. Reports record predictions/losses, labels, thresholds, source and
batch/state hashes, observed dependency versions and the workspace-lock hash.
CI installs the existing locked workspace environment; recording its lock hash
locally does not assert that a local environment was installed from that lock.

The existing nightly/PR command includes these tests and retains `learning.json`
under the `kumo-pytorch-aml-learning` artifact for 14 days, including available
reports on failure. For a focused local run with retained reports, select a new
directory (pytest clears an existing `--basetemp`):

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini \
  --confcutdir=test/kumo_pytorch \
  --basetemp=test/kumo_pytorch/.runs/learning-check \
  test/kumo_pytorch/model_checks/test_learning.py -q
```

These are controlled fixed-batch overfitting checks. They do not establish
unseen-example accuracy, recovery under new masks, full-data learning or loading
throughput. The existing retail learning checks remain in place; these checks
exercise the copied combined and PRAGMA consumers with AML inputs.

## History coverage sweeps

[History sweep checks](../model_checks/test_history_sweeps.py) exercise the actual
DiskGraph input path at **16, 128 and 1,024 events**. They cover nuFormer
involvement, nuFormer next-token, PRAGMA masked values and combined inputs:
three sizes × four consumers/objectives × five profiles = **60 cases**.

For each size `N`, two accounts have `N` and `N−1` historical transactions.
Fixtures mix incoming, outgoing and self-transfers, reverse physical row order,
and plant rows exactly at prediction time `T` and at `T+1`. Those future rows
must never enter histories or fitted vocabularies. History events are one second
apart, making the time-window boundaries independently countable.

| Profile | Source/selection | Expected retained events per account |
| --- | --- | --- |
| Full | One account-event table; entire strict history | `N`, `N−1` |
| Count limit | Most recent `N−3` events | `N−3`, `N−3` |
| Time window | Original prediction interval `[T−N/2, T)` | `N/2`, `N/2` |
| Two tables | Disjoint even/odd transaction partitions, chronologically merged | `N`, `N−1` |
| Two tables + window + limit | Same merge/window, then most recent `N/4` events | `N/4`, `N/4` |

These are explicit test variants; the published full-history declarations are
unchanged. Combined preserves its v2 graph execution clock. Its graph and twelve
aggregate features are checked against the complete source history even when
the sequence uses a smaller window or cap.

The existing source oracles verify exact encoded values and order, padding,
calendar/age values, event/field boundaries and target eligibility. A flat history
has `11 × retained_events − 1` tokens, including separators; next-token inputs
and targets each have one fewer. PRAGMA has six values per complete event and
keeps source-qualified field-name IDs across the two tables. Deliberately
swapping those source identities must fail the reference check. Fitted state
must remain unchanged during batch production and model input validation.

The shared next-token reconstruction helper is reused for supervised token
histories; the PRAGMA and involvement helpers gained explicit source/selection
references. Generic cap/merge algorithms remain covered by the existing package
tests. All new cases run under the existing nightly/PR model-check command.

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini \
  --confcutdir=test/kumo_pytorch \
  test/kumo_pytorch/model_checks/test_history_sweeps.py -q
```

All 60 cases passed. These results establish batch correctness and consumer
input compatibility on controlled AML-shaped fixtures. They do not establish
long-history neural forward/backward performance or complete-population AML
execution.

## Batching coverage sweeps

[Batching checks](../model_checks/test_batching_sweeps.py) add **23 cases** using
materialized Parquet anchors and DiskGraph. All five AML workload/model
combinations exercise **32, 128 and 512 samples** per batch. Each case delivers
two full batches and a five-sample tail, across native read chunks. Unique sample
IDs alternate between two accounts with six and seven historical transactions;
only two separate training anchors fit the encoders.

Eight additional cases apply token budgets to nuFormer involvement/next-token,
PRAGMA and combined inputs, with a 32-sample ceiling:

| History representation | Budgets | Independently checked cost |
| --- | --- | --- |
| Flat padded nuFormer/combined | 304 and 1,292 slots | Batch size × longest full token history, including padding |
| PRAGMA packed fields, padded event metadata | 168 and 714 slots | Sum of the six field values per selected event |

The smaller budget preserves request order; the larger enables length bucketing
and shuffling. Next-token planning budgets the full history before removing one
token per sample. Checks verify both this source-derived cost and actual emitted
tensor sizes. Budgets constrain the selected history input, not total graph or
model memory. Chunk boundaries and bucketing may leave spare budget capacity.

Every selected sample must arrive exactly once, with no training-split leakage
or marked padding. Existing task oracles check source values, masks, targets,
graph ownership/edges and combined count features against the delivered sample
IDs. Copied model input validators accept each batch; fitted state is unchanged.
These are input correctness checks, without large-batch neural training or
throughput claims. Generic planning and distributed padding remain owned by the
existing package tests; this is a selected matrix, not a history × batch-size
cross-product.

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini \
  --confcutdir=test/kumo_pytorch \
  test/kumo_pytorch/model_checks/test_batching_sweeps.py -q
```

All 23 cases passed. The existing nightly/PR model-check command automatically
includes them.

## Graph coverage sweeps

The [package graph sweeps](../../../packages/experimental/kumo-pytorch/tests/test_graph_sweeps.py)
reuse the existing conformance structure, edge uniqueness and point-in-time
oracles. **36 cases** cover fan-outs **2, 32 and 128**, one/two/three-hop paths,
two degree profiles and both existing engines independently. Six root entities
have either 192 eligible events each or degrees `[384, 3, 3, 3, 3, 3]`.
Distinct timestamps make recent-neighbor membership unambiguous. Reversed source
order, a row exactly at the inclusive cutoff and future rows exercise ordering
and boundary safety; the existing oracle also adds future rows and checks that
eligible neighborhoods are unchanged. These cases run in `package-regressions`.

The [AML graph sweeps](../model_checks/test_graph_sweeps.py) add **12 DiskGraph
cases**: GraphSAGE/combined × the same three fan-outs × two selected-root degree
profiles. The two accounts have 192/192 or 384/3 historical transactions, mixing
incoming, outgoing and self-transfers. Fan-out applies separately to sender and
receiver relations; the second hop still restores transaction endpoints.

An independent source selection takes the most recent eligible transactions
per role and verifies exact node membership, edges and sample ownership. The
v2 clock includes `T−1` and excludes `T`/`T+1`; future-only categories cannot
enter fitted vocabularies. Combined token histories and all twelve count
features retain complete history even when the graph is capped. A full-history
graph expectation deliberately fails for capped cases. Labels, fitted-state
stability and copied model input validation are also checked.

These are explicit test variants; published full-history AML contracts stay
unchanged. The AML checks validate inputs at these sizes, without large-graph
training or performance claims. “Uniform” describes root degree, not a uniform
random sampling strategy: all these cases use recent-neighbor sampling. Node
features retain existing coverage. **Edge attributes remain unresolved** because
the graph declaration exposes node fields, not edge fields. The generic
one-to-three-hop sweeps do not change AML's two-layer model binding.

From the package directory, run `python -m pytest tests/test_graph_sweeps.py
--require-imports -q`. From the repository root, run:

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini \
  --confcutdir=test/kumo_pytorch \
  test/kumo_pytorch/model_checks/test_graph_sweeps.py -q
```

## Feature and encoding coverage sweeps

The [package feature/encoding sweeps](../../../packages/experimental/kumo-pytorch/tests/test_feature_encoding_sweeps.py)
add **24 cases** through the existing `package-regressions` job:

| Dimension | Selected sizes | Checks |
| --- | --- | --- |
| Output feature width | 8, 32, 128 columns | Mixed aggregate/derived fields with raw, learned Z-score and explicit bucket encodings; exact field names, values and validity |
| Vocabulary cardinality | 16, 256, 4,096 categories per source field | Separate and shared learned state, exact frequency-ordered mappings, missing/unknown values and future-row exclusion |
| Controlled BPE text length | 16, 128, 1,024 tokens in the longest text | Exact token IDs and piece offsets, shorter/empty/null/unknown texts and validity |

Every case runs independently on both existing engines: DiskGraph model output
and the in-memory engine's supported inspectable output. No cross-engine batch
identity is required. Fitted-state fingerprints must remain unchanged during
delivery.

The width sweep reuses the existing reduction fixture and composes count,
distinct count, sum, min/max, mean, population variance/std, recency, grouped
count and derived offsets into wider output sets. Explicit answers cover two
source windows and exclude a future event. Basic arithmetic, mode, grouping and
null rules remain owned by existing tests; this does not reproduce each golden
under AML or establish wide physical source rows.

Vocabulary fields overlap halfway: shared state has 24/384/6,144 entries, while
separate state has 16/256/4,096 each. Future rows contain unseen values and nulls;
their serving anchors are outside the training split and encoder cutoff. The
text sweep reuses `C-ENCODING-003`'s supplied BPE state and known `red`/`bike` IDs.
It tests emitted text lengths, not BPE training vocabulary-size scaling.

All 24 cases passed. No AML workload/model, package runtime or workflow changes
were needed. From the package directory:

```sh
python -m pytest tests/test_feature_encoding_sweeps.py --require-imports -q
```

These are bounded CPU correctness sweeps. Physical wide rows and consumer empty
histories are covered separately below; source-row/fit-anchor scaling remains pending.

## Empty histories and wide source rows

[Twenty AML cases](../model_checks/test_empty_histories.py) cover all-empty and
mixed empty/nonempty histories across all five workload/model combinations on
both existing engines. Two available fixture accounts have no transactions;
encoders are fitted on separate nonempty training accounts. Independent source
and task checks verify input ownership, graph membership, sequences, aggregates
and target eligibility before executing the copied model's forward/backward pass.

Empty-history involvement samples retain their external labels and contribute
to binary loss. All-empty masked-value/next-token batches have no eligible
targets, finite zero loss and zero gradients; the caller must skip the optimizer
step when weight is zero. Mixed batches retain only eligible pretraining targets.
Emitted inputs/targets and fitted state must remain unchanged.

These tests exposed a copied nuFormer consumer bug on both engines: its internal
empty-history padding position produced one logit position against zero Kumo
targets. The local `forward_kumo` now returns logits/hidden states at the emitted
input width. Both formerly failing cases remain ordinary required regressions;
no skip, expected failure or target rewriting was introduced. The installed
package and Arena were not changed. See [model provenance](../models/README.md#copied-source).

[Twelve package cases](../../../packages/experimental/kumo-pytorch/tests/test_wide_source_rows.py)
cover **32/128/512 physical data columns**, plus the primary key, on both engines.
Each eight-row table mixes integers, floats and strings. Dense cases emit every
field; null-heavy cases project every seventh field, including an entirely null
column. Checks verify exact selected columns, values and validity, missing/unknown
category IDs, delivery across partial batches and unchanged fitted state. These
are physical Parquet column tests for DiskGraph, not merely many derived outputs.

Both modules run through the existing nightly/PR owners. Generic duplicate,
cutoff and null rules retain their existing package tests; earlier graph sweeps
already cover a selected 384-event hot root. This does not establish large-scale
hot-key behavior, memory limits or throughput.

From the package directory, run `python -m pytest tests/test_wide_source_rows.py
--require-imports -q`. From the repository root:

```sh
python -m pytest -c test/kumo_pytorch/pytest.ini \
  --confcutdir=test/kumo_pytorch \
  test/kumo_pytorch/model_checks/test_empty_histories.py -q
```

## Independent source and fitting-anchor sweeps

The [package source/anchor sweeps](../../../packages/experimental/kumo-pytorch/tests/test_source_anchor_sweeps.py)
cross **64, 1,024 and 16,384 event rows** with **8, 64 and 512 fitting anchors**:
nine cells on each existing engine, **18 cases** in total. Eight root rows and
five held-out anchors are additional to those counts. Changing fitting-anchor
count leaves the source unchanged; changing source size leaves fitting-anchor
count under its independent parameter.

These cases reuse the materialized-feature contract and anchor-table binding.
For each root, event times are `1..N`, with amounts `root × time`, so counts and
sums at any cutoff have exact independent answers. Reversed physical order,
inclusive cutoffs, full-history held-out anchors and an empty held-out window
exercise selection. Anchor IDs are unique; repeated root/cutoff contexts remain
separate fitting observations, so statistics must preserve their multiplicity.

Checks verify train-only Z-score mean/population scale and vocabulary membership,
then raw counts/sums, standardized values, vocabulary output, exact split delivery
and partial batches at size 31. Fitted state must stay unchanged while serving
train/validation/test. DiskGraph binds full Parquet source/anchor tables and uses
model output; the in-memory engine uses its supported row bindings and
inspectable output. No cross-engine batch identity is required.

All 18 cases passed, zero failed/skipped. The existing nightly/PR
`package-regressions` command collects the new module. From the package directory:

```sh
python -m pytest tests/test_source_anchor_sweeps.py --require-imports -q
```

This completes the selected bounded correctness matrix. It does **not** satisfy
the document's larger-than-RAM requirement: source/index memory, out-of-core
behavior, large hot-key workloads and full AML population execution still need
separate evidence. No timing or memory threshold is introduced here.

## Validation and current limits

The verified HI-small probe selected 17 roots, retaining 1,067 transactions,
2,102 account events and 129 endpoint accounts after scanning all 5,078,345 source
transactions. The full-history bound was 166 events. All five cases completed
one optimizer update and validation/test loss checks with four anchors per split.

This exposed an engine decimal transport bug: DiskGraph returns amounts as
strings before typed validation. The branch includes the existing fix from
commit `522e9ad` ([PR 348](https://github.com/NVIDIA/kumo-engine/pull/348)) and its
regression tests. It restores declared decimal arrays through safe casts and
rejects malformed, overflowing or lossy values. Source amounts are unchanged.

The existing nightly/PR workflow runs the tiny model, raw-file pipeline, source
mapping and anchor checks. Downloaded HI-small runs are currently invoked with
the command above; CI data access and storage setup remain pending. No full-data
job, GPU run or complete epoch is claimed. The 50–100-step checks above use only
the tiny controlled fixtures. Current AML suite: 352 passed, zero failed/skipped,
including the copied nuFormer fix and all 20 empty-history checks. All 12 new
physical wide-row cases passed separately. Earlier runs passed the 36 package
graph and 24 feature/encoding cases. The subsequent source/anchor increment
passed 18 new package cases; unchanged AML checks were not rerun. Downloaded
AML and the complete package/conformance/retail nightly were not rerun.
