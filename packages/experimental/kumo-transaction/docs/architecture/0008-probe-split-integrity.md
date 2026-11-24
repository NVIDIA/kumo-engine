# ADR 0008 — Probe split integrity and metric support

## Status

Accepted

## Context

`outputs/pragma_like/cpu_synth_1m/metrics.json` reported ROC-AUC and
PR-AUC of exactly **1.0** for all three `default_12m` probes, including the
`default_12m_raw` probe, which is five hand-written aggregates and no model.
A perfect score from a five-feature logistic regression is not a result; it is
a symptom.

The audit ruled out the two suspected causes and found a third:

- **Not the temporal cutoff.** `build_sequences_at_label_cutoffs` truncates
  each history at that row's `eval_ts`. Instrumenting every probe pair on the
  committed fixtures found zero events with `event_timestamp > eval_ts`, and
  `tests/unit/test_pragmatiq_adapter.py::test_eval_ts_cutoff_excludes_future_events`
  already pins that behavior.
- **Not label-derived features.** `raw_feature_vector` is mean amount, max
  amount, event count, and the last event's hour and day-of-week, all computed
  from events already truncated to the cutoff.
- **It is entity overlap in the probe split.** `_split_label_pairs` divided
  labeled entities 70/30 per class, but its `split_ids` helper returned
  `ids, ids` when a class held exactly one entity. That entity was therefore
  placed in **both** the probe training set and the probe evaluation set. The
  probe was scored on a row it had been fit on.

A single positive entity is the normal regime for these runs, so the path was
routinely taken. It is also self-verifying: with one positive and
`class_weight="balanced"`, the fitted probe ranks that memorized row first, so
ROC-AUC is 1.0, PR-AUC is 1.0, recall@k is 1.0, and precision@k is exactly
`1/k`. The recorded artifact reads `precision_at_k: 0.2` at `k = 5`. That is
the signature, not a coincidence.

This is leakage, not a small-sample artifact. The metric was unearned rather
than merely noisy. Repository rules forbid performance claims without
reproducible evidence, and a leaked 1.0 in a committed artifact is the
strongest possible false claim.

A second, independent problem made the first one invisible: a probe report
carried no support counts, so `roc_auc: 1.0` over one positive row was
indistinguishable in `metrics.json` from `roc_auc: 1.0` over a real test set.

## Decision

1. **Probe splits are entity-disjoint, always.** Entity-level train/evaluation
   splits live in `tfm.data.splits.entity.split_entities`, beside the temporal
   split, and are the only supported way to divide labeled entities for a
   probe. The function guarantees the two sides share no entity.
2. **A class too small to split is not scored.** `split_entities` requires
   both classes to be present with at least two entities each. Below that, it
   raises `InsufficientLabelSupportError`. The caller skips that task with an
   explicit warning; it never emits a leaked number and never falls back to a
   random split. A missing probe is an honest result. A perfect one is not.
3. **Every probe report carries its own support.** `classification_metrics`
   reports `n_eval` and `n_eval_positive` alongside the scores. A reader
   cannot see a metric without seeing how many rows produced it.
4. **Metric tables mark unsupported rows.** `metrics_markdown` flags any row
   with fewer than `MIN_TRUSTWORTHY_POSITIVES` positives, so a rendered report
   cannot present an under-supported score as a finding.
5. **Contract tests enforce disjointness**, in `tests/contract/`, for both the
   fraud and forecast probe paths.

   Correction: as first written, the tests covered the forecast path only, and
   the fraud probe scored the same entities at two cutoffs.
   [ADR 0013](0013-fraud-probe-split-integrity.md) (accepted) made the fraud
   probe entity-disjoint with per-side label windows, and
   `tests/contract/test_fraud_probe_windows.py` now covers it.

## Consequences

- `default_12m` probes disappear from the metrics of the smallest fixtures
  instead of reporting 1.0. This is the intended behavior: those fixtures never
  had enough labeled entities to support a probe.
- `metrics.json` gains `n_eval` and `n_eval_positive` on every probe entry.
  This is an additive change to a reported artifact, not to a batch, model
  output, checkpoint, or configuration contract.
- Recipes that want scored forecast probes must supply a book with at least two
  entities per class. The PRAGMA size-ladder configs and the PRAGMA comparison
  criteria in the roadmap already require far more than that.
- The stale leaked artifact under `outputs/pragma_like/` is deleted
  rather than kept for comparison.
