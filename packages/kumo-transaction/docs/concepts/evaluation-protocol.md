# Evaluation protocol

Every recipe answers one question per task: does a frozen TFM embedding add to
a raw-feature XGBoost on the same rows? This page defines the arms that answer
it, the metrics every arm reports, how lift is stated, and what a recorded
result must contain. The split and label rules the arms rely on are in
[leakage and temporal splits](leakage-and-temporal-splits.md).

## The headline

Per task, a run reports
([ADR 0018](../architecture/0018-one-headline-comparison.md)):

- `<task>_xgboost_raw`: XGBoost on the row's raw fields only, which are the last
  event's own fields (for an event-level task, the labelled transaction) and
  the profile. No count, recency, or aggregate of earlier events. This is
  Blueprint v1's raw arm.
- `<task>_xgboost_raw_embedding`: the same row with the frozen embedding
  appended. The **headline** is its lift over `xgboost_raw`.
- `<task>_tabular_xgboost`: the reference. It uses hand-built aggregates of the
  same truncated history, with no encoder. An embedding that only matches it
  has learned what a feature engineer would write by hand.
  `tabular_xgboost_embedding` appends the embedding to those aggregates.

The model card opens with this table. The history-window task adds
`xgboost_raw_embedding:single_event`, which pairs the same raw row with the
transaction-alone embedding. It differs from the headline only in whether the
encoder saw the history. With `evaluation.surprise`,
`xgboost_raw_embedding:surprise` also appends the model's next-event surprise
on the row's latest events.

Without the `baselines` extra (CPU smoke configs, the slot ablations), the
XGBoost arms are off and a logistic stack runs instead. `raw` is five
per-sequence aggregates, `embedding` is the frozen embedding, and `combined`
is both. It shows the wiring; it is not evidence.

## Fitting the XGBoost arms

Every XGBoost model comes from one constructor,
`tfm.evaluation.tabular.xgboost_classifier`, on CPU or with
`tabular_baselines.device: cuda`. There are two protocols, and a config sets
exactly one:

- `xgboost_params`: one fixed setting shared by every arm. Early stopping on an
  80/20 entity-disjoint split of the train side picks only the tree count, so
  arms differ only in their features. The reference recipes use this, with the
  median of a recorded search's selections.
- `xgboost_trials`: an Optuna TPE search per arm that maximizes PR-AUC on the
  same inner split, refitted on the whole train side. A search per arm moves
  single scores by a few points between arms that see the same signal.

The evaluation side is never seen while fitting. The selected setting is
written to the probe manifest under `tabular_baselines`.

The tabular features are built per key from the event contract, naming no
dataset-specific field, so they apply to every adapter:

- counts, recency, and inter-event gaps;
- statistics of each numeric field;
- shares of each categorical field's most frequent train-side values;
- how often the last event's category or text value appeared earlier in the
  history, and how many new values arrived in the last 1, 7, and 30 days;
- the profile attributes, and milestones dated by the cutoff, that the
  key-value-time tokenizer also encodes.

`exclude_keys` names identifiers, such as a user id, that no feature is built
from. Otherwise a tree memorizes them per entity.

## Metrics

Every arm reports `roc_auc`, `pr_auc` (average precision), `precision_at_k`,
`recall_at_k`, `calibration` (expected calibration error), `precision` and
`recall` at 0.5, and their support, `n_eval` and `n_eval_positive`. A task
whose evaluation side has fewer than five positives is flagged as
unsupported, and a side with a single class scores `roc_auc` 0.5 by
definition. Neither is a result.

Ranking, calibration, and efficiency are separate claims; never combine them
into one score.

## Lift

**Lift** is the relative change of a metric over a reference arm,
\(x / x_{\text{ref}} - 1\), computed per seed on the same rows. A record
states which reference it uses. Lift over `xgboost_raw` is the headline. Lift
over the five-aggregate logistic `raw` is a weaker claim, because no team
would deploy that baseline.

Because an embedding can carry the aggregates a feature engineer would write,
a record also states the **gain recovered** on PR-AUC,
\((\text{raw+emb} - \text{raw}) / (\text{tabular} - \text{raw})\): the share
of the hand-built features' gain that the embedding recovers without them.
`benchmarks/lift_table.py` prints both. PRAGMA (arXiv 2604.08649) reports
PR-AUC lift over production baselines; `tabular_xgboost` is this kit's
stand-in for one.

## Recording a result

A claim needs a record under `benchmarks/results/`
([index](../../benchmarks/results/README.md)): a summary `.md` and its
`.json`, produced by a committed config. The record states:

- the config, any overrides, and the seeds;
- the commit, and whether the working tree was clean;
- the dataset, its sample, and the split;
- the hardware, precision, batch size, sequence length, and wall time;
- per task, the headline, the reference, and the positive count, each over
  every seed;
- what the run does not show.

A lift is a claim only with its range over seeds (`benchmarks/seed_range.py`,
at least three, each moving pretraining and the probe split) and its positive
count. Smoke runs never count. Optimized kernels are compared with the eager
baseline on latency, throughput, and peak memory.

Do not treat synthetic-data metrics as evidence of production fraud
performance.

## Experimental graph recipe comparisons

The graph recipe family currently uses the CPU-safe logistic comparison, not
the XGBoost headline protocol. It retains the same raw five-history-aggregate,
embedding, and combined arms on identical probe rows, plus a separately named
graph-tabular arm (counts and sums of sampled input-node features). A sequence-only
binary control trains and selects on exactly the hybrid's history/label batches,
with identical initial shared-stack parameters, seed, training settings and
validation rows. FutureSubgraph mode additionally retains the historical
next-event control trained on the same pretraining windows. Both use the same
probe rows and history lengths; the historical arm does not isolate the effect
of graph conditioning from downstream supervision. The hybrid additionally
probes its exact frozen GNN. Graph and sequence views share sample IDs/cutoffs;
no cross-dataset comparison or quality claim is implied.

Select a directly supervised GNN using sample-weighted binary validation loss,
or a FutureSubgraph GNN using valid-entry-weighted self-supervised validation
MSE. Select the hybrid using sample-weighted binary validation loss, before test
probes. The historical next-event control weights validation loss by the number
of target tokens selected by both token and event masks, matching its objective.
Longer histories contribute their additional targets rather than one target per
entity. `task_head` reports the selected binary head on held-out test owners
when the main model is supervised; the logistic probe arms remain separate.
The task is any available fraud in the truncated history, using existing
sequence-label semantics. It is not a future-fraud forecast. Checkpoints record
selection steps, root-owner splits, and the cross-owner graph-context policy.


Graph training can share same-cutoff neighborhoods; validation/test and embedding
extraction stay disjoint. Report the sharing policy because BatchNorm weights
unique versus repeated nodes differently. The three-arm Engine-shared,
Engine-disjoint, and historical-custom-disjoint timing comparison isolates
architecture from graph policy at a fixed-batch level; it does not establish
pure runtime equivalence or downstream quality. Preparation/transfer are
reported separately from forward/backward/optimizer timing. No full training
campaign is justified solely by a small synthetic timing comparison.


Hop trimming and the attention kernel are execution conditions, recorded with
the objective/sharing mode. The historical L4 comparison predates trimming and
input ReLU removal and cannot be reported as current throughput. Trimmed versus
untrimmed root parity is checked in evaluation; training BatchNorm populations
may differ. SDPA/eager output and gradient equivalence is checked separately.


Matched binary control metrics use `sequence_supervised_*`, including its trained
`task_head`. Historical next-event embeddings retain `sequence_only_*` in
FutureSubgraph mode; in binary mode that legacy prefix is an alias for the
binary control. Sequence controls consume history-only batches and share the
Blueprint v2 encoding implementation. Coverage guards reject absent fraud
sources or owners with no available source labels by cutoff; they preserve the
existing truncated-history any-available-positive rule.


For the five per-sequence logistic raw features, omitted, NaN and infinite
amounts contribute zero to the mean and maximum. Every retained event still
counts in the mean denominator. This is the existing missing-amount convention,
including when finite and missing values share a history; it does not change
the split, labels or the configured amount field.
