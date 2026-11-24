# Data contract

A **transaction event** is the atomic record: synthetic `entity_id`, `event_id`,
UTC `event_timestamp`, categorical/numeric/text fields, and optional labels.

Labels are not features. Each `EventLabel` has a `level` (`event`, `entity`, or
`sequence`) and `available_at` time. Model inputs must not include fraud,
chargeback, or other post-outcome fields.

A **transaction sequence** is the ordered events for one entity with timestamp
`<= cutoff`, optionally clipped to an observation window, truncated to a maximum
event count. Core-surface sequences are causal.

Optional **entity profiles** (`EntityProfile`: static attributes plus dated
lifelong milestones) and **transfer edges** are domain types for adapters. They
are not consumed by the flat causal decoder.

## External parquet: pragmatiq

`tfm.data.adapters.pragmatiq` maps the dynamiq-ai/pragmatiq four-file layout
(`events.parquet`, `profiles.parquet`, `transfers.parquet`, `labels/*.parquet`)
into these domain types. `user_id` becomes `entity_id`; `event_id` is
synthesized from `(user_id, ts, index)`; `source` is stored on
`categorical["source"]`; string maps are split into numeric/categorical/text.

The 12-field `flat_fields` tokenizer only represents **transaction** events
after `events_for_flat_fields` aliases keys such as `merchant` →
`merchant_name`. App, trading, and communication events are dropped with a
warning. Heterogeneous sources need the key-value-time tokenizer.

Forecast label tables carry `eval_ts`. Recipe `pragma_like` rebuilds
each `default_12m` history with cutoff equal to that row's `eval_ts` so events
after the evaluation point are not encoded. MLM pretraining still uses the
recipe temporal split. Probe train/test is a seeded entity split of labeled
rows (parquet books do not ship predefined folds).

## External parquet: TabFormer

`tfm.data.adapters.tabformer` maps IBM TabFormer columns (`User`, `Amount`,
`Use Chip`, `Merchant Name`, `MCC`, `Is Fraud?`, ...) into `TransactionEvent`.
Resolve the book with `data.path` or `TFM_TABFORMER_ROOT`. Prefer
`temporal_split/*.parquet` when present (`split_protocol: precomputed` is the
v1 80/10/10 cumulative split, documented as an exception to timestamp cutoffs).
Entity subsample (`max_entities`) prefers users with at least one fraud label
on train/test parquet. Sequence-level fraud is **any available positive in the
history window**, not the last event only (split-end last events are almost
never fraud at TabFormer's ~0.1% rate).

Do not store real customer, account, card, or merchant identifiers. Use
synthetic IDs in fixtures and tests.
