# ADR 0013 — Fraud probe labels and entities are disjoint across sides

## Status

Accepted. Amends decision 5 of [ADR 0008](0008-probe-split-integrity.md),
which holds for the forecast probes but not for the fraud probe. Implemented in
`tfm.data.labels.windows` (`LabelWindow`) and `tfm.runtime.evaluate`, which
every stack now shares; `tests/contract/test_fraud_probe_windows.py` and
`tests/contract/test_probe_split_integrity.py` hold decision 5.
[ADR 0015](0015-dataset-strategy-and-spliced-fraud.md) applies these windows
to the spliced identity-theft task and sets the pragmatiq cutoffs from the
label calendar.

## Context

ADR 0008 made forecast probes (`default_12m`, `churn_6m`, and the rest)
entity-disjoint through `split_entities`. The fraud probe does not use that
path. Both runtimes score it the same way:

- `train`: one sequence per entity at the train cutoff, labelled 1 if any event
  in the sequence carries an available fraud label (`sequence_fraud_label`).
- `test`: one sequence per entity at the test cutoff, labelled the same way.
- `evaluate_binary_probes(train, test)` fits the raw, embedding, and combined
  probes on the first and scores them on the second. The key-value-time LoRA
  stage (`run_kvt_lora`) fine-tunes on the same pair.

That is an out-of-time protocol over the **same entities**, and each label
looks at the **whole history window**, not at the side's own period. A
pre-scale audit of `blueprint_v2/config.scale.yaml` on the
Generator v2 book (5,000 users, seed 42, 256-event cap, 730-day window)
measured the overlap. The audit wrapped the sequence builders
`prepare_kvt_data` calls, so these are the sequences the recipe scores:

| | count |
|---|---|
| entities on both sides | 5,000 of 5,000 |
| eval-positive entities | 182 |
| of those, also train-positive | 34 |
| labelled fraud events inside both a train and an eval sequence | 311 |
| events shared by train and eval sequences | 248,627 |
| earliest event in an eval sequence | 2023-02-02 (test cutoff 2025-01-31) |

The 256-event cap reaches back past the train cutoff for users with sparse
activity, so the same fraud event makes a user positive on both sides. The
probe is then partly scored on labels it was fit on, and partly on a user's
recurring propensity rather than on anything the embedding encodes about the
eval period. The encoder was also pretrained on the train-side sequences, so
those shared events were in its pretraining corpus as well.

Nothing else in the audit failed. Every sequence's events are at or before its
cutoff and in order; labels are counted only once available; no event or
profile input is label-derived; post-cutoff profile milestones are dropped.
This is a probe-protocol defect, not an input leak.

Two related findings from the same audit are recorded here because they bear
on the same criterion, roadmap Milestone 2 criterion 7 ("no open leakage
questions"):

- **Forecast rows dated before the pretraining cutoff.** Every `default_12m`
  row in the Generator v2 book has `eval_ts = 2024-02-01`, before
  `train_end = 2024-06-01`. Probe histories are cut at `eval_ts`, but the
  encoder was pretrained on those users' events from 2024-02-01 to 2024-06-01,
  which fall inside the 12-month label horizon.
- **Thin support.** `aml` had 3 positives at 5,000 users (2 train, 1 eval).
  ADR 0008 already flags rows under `MIN_TRUSTWORTHY_POSITIVES` in
  `metrics.md`; no change is proposed for it.

## Decision

1. **Each side's label comes from its own period.** A fraud-probe row is
   positive only if a fraud event inside that side's label window carries an
   available label. Windows are half-open, `(start, end]`, so an event on a
   boundary belongs to exactly one side: train is
   `(train_end - observation_window, train_end]`, validation is
   `(train_end, validation_end]`, and test is `(validation_end, test_cutoff]`.
   Events before the window stay in the history as inputs; their labels do not count.
   The label window is part of the label contract: it is passed explicitly
   (not inferred from the sequence) and written to `split_manifest.json`
   beside the cutoff and label delay.
2. **The fraud probe is entity-disjoint.** Labelled entities are divided with
   `split_entities`, stratified by the eval-side label. The probe fits on the
   train-side rows of the train entities and is scored on the eval-side rows
   of the evaluation entities. Too few entities per class skips the probe with
   a warning, as in ADR 0008.
3. **Fine-tuning follows the same split.** The `fraud_task.finetune` arms use
   the rows and labels of decision 2, so the fine-tuned and frozen arms stay
   comparable.
4. **Forecast rows must postdate pretraining.** A forecast probe is scored only
   if every row's `eval_ts` is at or after the pretraining cutoff
   (`train_end`). Otherwise the task is skipped with a warning naming both
   dates.
5. **Contract tests.** For every stack: no entity on both fraud-probe sides;
   no labelled event counted by both sides; every counted label inside its
   side's window; forecast probes refused when rows predate `train_end`. Each
   test is shown to fail on the current protocol before the change lands.
6. **ADR 0008 decision 5 is corrected** to say which paths its tests cover.

## Resolved questions

1. **`default_12m` on the Generator v2 book is skipped.** Decision 4 applies
   as written. Moving `train_end` back to 2024-02-01 would leave 13 of the
   book's 25 months for pretraining, and reporting a pretraining-overlapped
   probe is the kind of number the repository rules forbid.    PRAGMA criterion 6
   is met by a config whose `train_end` precedes the book's `eval_ts`, not by
   relaxing the rule.

   *Update, 2026-09-26:* `configs/data/pragmatiq/world.yaml`
   generates a 31-month book whose `default_12m` labels are issued on
   2024-07-01, after `train_end`. On that book the task is scored under
   decision 4 as written.
2. **No same-entity arm.** An out-of-time, same-entity probe answers a
   legitimate question, but a second fraud number beside the headline invites
   quoting the easier one. It can return as its own ADR if a recipe needs it.

## Consequences

- SDK-native fraud metrics change for `decoder_fraud_baseline`,
  `blueprint_v1`, `pragma_like`, and `blueprint_v2`:
  eval sets shrink to about 30% of entities, and positives are only
  eval-period fraud. No recorded result depends on the current protocol: both
  records then under `benchmarks/results/` used v1's own probe (`v1_probe.py`,
  since archived; ADR 0017), which this ADR does not touch.
- Small fixtures may lose their fraud probe, as ADR 0008 did for
  `default_12m`. Smoke tests that assert a fraud metric need enough entities
  per class or must assert the documented skip.
- `sequence_fraud_label` gains an explicit label-window argument. It is not
  part of the core export surface. Labels are now attached only by
  `collate_batch` in `tfm.data.collate`; tokenizers no longer attach them.
- Roadmap Milestone 2 criterion 6 (the history-window probe) is complementary:
  it changes what one row is, not how rows are divided, and inherits these
  rules. It is implemented as `evaluation.history_window`.

## Alternatives rejected

- **Event-disjoint labels only, same entities.** Removes the double-counted
  events but still scores a probe on users whose fraud propensity it was fit
  on. See resolved question 2.
- **Random row split of a single table.** Mixes cutoffs and reintroduces
  ADR 0008's defect.
- **Document the overlap and keep the protocol.** A leaked fraud metric is the
  claim the repository rules forbid.
