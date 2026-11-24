# ADR 0018 — One headline comparison per task

## Status

Accepted. Amends how [ADR 0013](0013-fraud-probe-split-integrity.md) and
[ADR 0017](0017-v1-blueprint-on-the-sdk.md) arms are reported; changes none of
their split or label rules.

## Context

A full run scored every task three ways: a logistic `raw` / `embedding` /
`combined` stack over five per-sequence aggregates, the tabular arms
(`tabular_logistic`, `tabular_xgboost`, `tabular_xgboost_embedding`), and the
XGBoost arms `xgboost_raw` and `xgboost_raw_embedding`. A `metrics.json` held
up to a dozen rows per task, from three classifiers, over three different
feature sets. Readers compared rows that differ in both features and
classifier. The milestones ask one question: does a frozen TFM embedding add
to a raw-feature XGBoost baseline on the same rows?

Two smaller defects came with it. The probe's optional XGBoost head built its
classifier separately from the tabular arms, with other defaults. The
history-window probe's `single_event` control reused the history row's raw
features, so its logistic `single_event_raw` arm repeated
`history_window_raw` under a name that suggests event-only features.

## Decision

1. **Headline.** Per task, `<task>_xgboost_raw_embedding` over
   `<task>_xgboost_raw`: the same raw row (the last event plus the profile,
   Blueprint v1's raw arm), the same classifier, and only the embedding added.
   The model card opens with this table, lift included.
2. **Reference.** `<task>_tabular_xgboost`, hand-built aggregates of the same
   history with no encoder, is reported beside it. An embedding that only
   matches it has learned what a feature engineer would write by hand.
3. **One XGBoost constructor.** `tfm.evaluation.tabular.xgboost_classifier`
   builds every XGBoost model, CPU or `device: cuda`.
4. **The logistic stack is the CPU smoke path.** It runs only when the
   XGBoost arms are off, so smoke configs need no extras. With them on, a run
   reports the XGBoost and tabular arms only.
5. **The single-event control** reports `single_event_embedding` and
   `single_event_combined`, not `single_event_raw`. In the XGBoost arms it is
   `history_window_xgboost_raw_embedding:single_event`: the history row's raw
   features plus the transaction-alone embedding, so it differs from the
   headline only in what the encoder saw.

## Consequences

- Runs with the XGBoost arms no longer write `raw`, `embedding`, or
  `combined`. Records made before this ADR keep them; the results index notes
  which records predate it.
- The slot ablations (`benchmarks/ablations/building_blocks`) still compare
  logistic arms on CPU, because their configs run without the extras.
- A lift is a claim only with its range over seeds and its positive count, as
  [the evaluation protocol](../concepts/evaluation-protocol.md) requires.
