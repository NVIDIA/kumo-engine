# Journeys

Two independent axes. A design can satisfy one and fail the other, so they are
evaluated separately.

## User journeys

Whether a customer gets correct data without help from their IT department.

| | journey | satisfied by | evaluated by |
|---|---|---|---|
| U1 | Reads a table holding only the access they already have | mode chosen from probed grants | live conformance, minimal-privilege credential |
| U2 | Re-runs with unchanged data and pays nothing | `data_version` | `test_identity` + live re-read |
| U3 | Re-runs after data changed and gets the new data | identity changes with the data | live conformance |
| U4 | Reads a view, and is never served stale rows | the `hour_version` fallback | `test_identity` + live view read |
| U5 | Never configures a read mode | the first of `read_modes` | live conformance across vendors |
| U6 | Learns that a lapsed credential needs reconnecting | error taxonomy | `test_errors` |
| U7 | Gives credentials once in cloud, or never in local | caller resolves credentials; readers take a client | adapter tests |

U1 is the load-bearing one: a read-only analyst on Snowflake, a PAT holder on
Databricks, a `gcloud`-authenticated human on BigQuery. Any journey that needs
a grant they cannot self-serve has failed.

## Engineer journeys

Whether the code stays correct as it changes.

| | journey | satisfied by | evaluated by |
|---|---|---|---|
| E1 | Adds a warehouse by implementing one protocol | `SourceReader` | new package passes conformance unmodified |
| E2 | Diagnoses a failure from the error alone | errors name vendor, table, mode | `test_errors` |
| E3 | Changes shared code and learns what broke | one suite over every vendor | conformance runs per vendor |
| E4 | Runs the suite with no warehouse credentials | tiering | unit tier passes, live tier skips loudly |
| E5 | Reviews a vendor package for isolation | boundary tests | AST import check per package |

E4 matters because a suite that silently skips is worse than one that fails:
a green run must never be ambiguous about what it covered.

## Test tiers

- **unit** — no network. Contracts, semantics, validation.
- **conformance** — one suite every reader passes, run per vendor.
- **live** — real warehouses, real credentials, row-verified against
  `COUNT(*)`. Skips are reported, never silent.

Live tiers assert `rows == truth` *and* `truth > 0`; an equality that holds
because both sides are zero has tested nothing.
