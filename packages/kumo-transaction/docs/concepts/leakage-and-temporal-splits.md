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

## Graph inputs and future-subgraph labels

The experimental graph recipes enforce these input and target boundaries:

- Encoder rows, links, and features must be available by the anchor cutoff.
  Mutable untimed snapshots are refused; immutable identity roots are explicit.
- In optional FutureSubgraph mode, label-only membership at `t+H` is a separate graph. Terminal event times lie
  in `(t,t+H]`; later links cannot introduce membership. Late terminal source
  availability delays the target. A full horizon must lie inside declared source
  coverage and its availability inside the pretraining partition.
- Feature codecs fit only training owners: at the earliest historical anchor
  for FutureSubgraph, or the task cutoff for direct supervised training. Future
  target selection/scaling uses valid self-supervised training entries only.
  Direct supervised mode constructs no future targets or statistics. Validation
  reuses fitted transforms.
- Graph/history alignment uses complete sample identities, including cutoff.
  Repeated entities at different cutoffs have disjoint graph instances.
- A graph at history end is valid for a cutoff-level embedding but can leak
  future information to earlier next-event positions. `GNNSequence` therefore
  refuses within-history next-event objectives and surprise.
- Probe owner splits do not imply graph-context isolation. The graph recipes
  permit cutoff-bounded cross-owner transfer context and record that policy.

- Shared training components require identical cutoff and observation windows.
  A node cannot bridge earlier and later anchors. Repeated same-time roots may
  share nodes, while prediction labels retain sample order. Validation and
  embedding/probe inference always use separate graphs.


Graph observation coverage is the explicit `observed_through` source watermark,
not the timestamp of the last event. Absence of activity before that boundary
is allowed; it does not by itself make coverage incomplete. Probe cutoffs must
remain within coverage, and future targets still require their entire horizon
and source availability before the partition/watermark boundary.


Graph ingestion applies shared event-schema validation in every stage, including
FutureSubgraph pretraining without a tokenizer. Label-like categorical/numeric/text
fields remain prohibited even if they are not the selected amount field.
The selected amount field must have at least one finite value on a
training-owner event available to a graph-training input cutoff. For FutureSubgraph this excludes SSL validation
owners and events after the latest training anchor; target-window amounts alone
do not satisfy the requirement. Individual events may still omit the field.
