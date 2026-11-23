# Leakage and temporal splits

## Rules

1. Sort events by `(event_timestamp, event_id)`.
2. A sequence at cutoff `C` contains only events with timestamp `<= C`.
3. A label may be used at `C` only if `available_at <= C`.
4. Train / validation / test partitions are defined by `train_end` and
   `validation_end` on event time, not random row splits.
5. Downstream classifiers used in recipes are fit on the train partition only.
6. Fitted tokenizers (key-value-time and any data-dependent vocab) are fit on
   **train-partition sequences only**. The 12-field tokenizer is
   data-independent.
7. Forecast probes (`default_12m`) rebuild each history with cutoff equal to
   that row's `eval_ts`. Do not reuse MLM sequences whose cutoff is
   `train_end` / `validation_end` when those timestamps differ from `eval_ts`.
8. `split_protocol: precomputed` (TabFormer parquet partitions) uses each
   partition's last event time as the sequence cutoff, not YAML `train_end`.
9. Profiles are cut at the sequence cutoff, not the snapshot's `as_of`, which
   can be later. Lifelong milestones dated after the cutoff are dropped, and
   the rest are timed from the cutoff. Undated static attributes must not
   depend on events after the cutoff.
10. Time features fed to a causal model depend only on past events.
    `KeyValueTimeBatch.time_log` counts back from the sequence's last event and
    is for bidirectional encoders; causal decoders use `elapsed_log`.

## Prohibited

- Future events as inputs
- Chargeback-resolution or label-derived fields as features
- Fitting tokenizers or scalers on test-period rows when those objects affect
  train-time features (the 12-field tokenizer is data-independent)

If a change might alter these semantics, update contract tests and this document
in the same merge request.
