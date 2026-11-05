---
name: kumo-custom-populations
description: Train with caller-supplied labels, weights and splits or score exact entity/anchor rows through Snowflake source tables using the local Kumo API. Use when generated populations do not represent the requested rows.
---

# Caller-supplied populations

Preserve the query, graph and evaluation protocol. Read
[the population contract](references/populations.md) before creating a supplied table.
Snowflake sources stage local Parquet for this API. Do not switch to another connector or
claim that a local bundle is a source table.

Resolve the supplied source table and required column semantics. These constructors consume
existing tables; they do not authorize creating labels, renaming data, cleaning nulls or
changing splits. Use a user-approved prepared table when the source lacks required fields.

Validate schema, row types, keys, timestamps, labels, weights and splits and report the exact
bad field. Do not silently drop rows, cast IDs, normalize weights or replace manual splits —
not in a read-only query either: a field that needs changing is reported and asked about.

Before any code, reply with the plan and stop for confirmation: the table and its fields, the
weight and split columns, what you validated, the constructor and that it reuses the confirmed
`pq` and `snapshot`, and any shape the host adapter says is unverified there. A bad field gets
the ask-back instead.
Confirm the snapshot used to create the population and the snapshot passed to fit/predict
belong to the intended graph. Rebuild populations after incompatible graph changes.
