# ADR 0004 — Temporal evaluation

## Status

Accepted

## Decision

Temporal correctness is a repository invariant. Sequences include only events
with `event_timestamp <= cutoff`. Labels are usable only when
`available_at <= cutoff`. Train/validation/test partitions are defined by
explicit timestamps. Contract tests enforce these rules. Feature fields that
look like labels (`is_fraud`, `chargeback`, …) are rejected as model inputs.
