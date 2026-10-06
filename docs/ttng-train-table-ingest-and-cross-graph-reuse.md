# TT-NG: one train-table path, orphan-safe cross-graph ingest

**Companion:** [DiskGraph architecture and migration](../packages/diskgraph/docs/tt-ng-architecture-and-migration.md) (index
architecture), the contract docs on `diskgraph_ingest::ingest_stage_extension`
(orphan semantics) and `tt_executor::run_gen_bundle` (gen-bundle core).

## The problem this closes

A train table (TT) is generated against one graph snapshot, but
`trainer.fit(train_table, graph)` can bind training to a **different**
snapshot: the per-job `train_tables/<tt_id>/` extension is not part of the
graph snapshot, so the training pod re-ingests the TT's durable staging bundle
into whatever index the job points at. TT ingest resolves seed foreign keys
through a minimal-perfect-hash pkmap with **no membership verification** — a
seed FK value absent from the training graph did not miss, it silently
collided onto an arbitrary in-range node, and training completed on mismatched
labels with zero error signal.

The ingest contract: stop pretending "TT graph == training
graph". After gen, treat **every** TT — PQL-generated or customer-provided —
as a custom TT: ingest the bundle into the *training* graph, and on any
orphan FK **skip the unresolved reference and count the lookup or membership
failure**, never collide. In FK-free link-prediction resolution, null or
unparseable inputs can be skipped without incrementing those counters.

## The one-path model

```
PQL:    GenerateTrainingTableWorkflow ──► execute_tt_gen(plan)      ──┐
                                            └── run_gen_bundle        │  durable bundle
CUSTOM: CustomTrainTableWorkflow      ──► import ► validate ►         │  (tt.parquet +
          build_ttng_custom_train_table_bundle ──► custom gen entry          ──┤   tt_ingest_spec.json
                                            └── run_gen_bundle (same) │   + stage seeds)
                                                                      ▼
TRAINING JOB (any graph G, incl. G ≠ gen graph):
  materialize_training_task ► ensure_stage_ingested ► execute_tt_ingest(bundle, G)
      • every seed FK membership-verified against G's node data
      • orphans skipped + counted (never mis-resolved); counters in result + manifest
      • kumo warns "X% skipped", escalates ≥50%, optional KUMO_TTNG_ORPHAN_FAIL_PCT
  ► training samples from train_tables/<tt_id>/ under G
```

- **Gen** differs only in the producer. PQL runs the plan against the graph;
  custom reads every output column from the customer parquet
  (`CustomTtSource`). Both run the same DG gen core
  (`tt_executor::run_gen_bundle`): ttgen pass + `tt_ingest_spec.json` beside
  `tt.parquet`, emitting one self-describing durable bundle. Producers publish
  immutable bundles under
  `<logical_output_path>.ttng_staging/_bundles/<bundle_id>/`. Consumers receive
  the selected bundle’s physical `tt.parquet` path as `train_table_path` and
  derive its bundle directory from that file’s parent.
- **`materialize_training_table`** is a validation gate on TT-NG: it asserts
  the durable bundle is complete and early-returns. No ingest happens there
  (trip-wired by `test_materialize_training_task_ttng.py`).
- **The sole ingest** is `materialize_training_task` →
  `_ttng_ensure_train_table_extension` → `ensure_stage_ingested` →
  `execute_tt_ingest(staging_dir, index_path=G, tt_id)`. Present-and-complete
  extensions are reused; anything else is (re-)ingested from the bundle.
- The predict path uses the identical bundle + ingest machinery with a single
  `predict` stage.

## Orphan safety (the correctness guard)

`execute_tt_ingest` membership-verifies every external-parent FK resolution
against the training graph's node data (the resolved node's stored primary
key must equal the reference value — `TableReader::validate_pk_at_node`):

- **FK-edge path** (node prediction / forecasting / predict tables): a
  verify-mismatch skips the seed **edge** exactly like a genuine pkmap miss;
  the seed row is kept as a node.
- **FK-free resolution mode** (link prediction): an unresolved entity anchor
  drops the entire row. Orphan target-list elements are dropped; if a
  nonempty positive-target list loses every target, the row is dropped too.
  Source-empty lists remain, and exclusion lists do not cause row drops.
  Genuine pkmap misses and membership failures are counted.
- A parent table without rowpack PK data cannot be verified → the pkmap hit
  is trusted (**never over-drop**).
- Base-graph ingest is untouched: verification is scoped to
  `external_parent` relations and only constructed when a parent index
  exists, so normal graph ingest pays nothing.
- Temporal validity is deliberately **not** re-checked at ingest: the split
  and seed times are part of the TT contract the user asserts when reusing a
  TT (see "residual risks" in the design plan); membership is the
  correctness guard.

Counters flow as `{parse,lookup,membership}_failures` (+ per-relation
`by_context`, + samples) through `IngestStats` into both the
`execute_tt_ingest` result dict and the persisted
`train_tables/<tt_id>/diskgraph_manifest.json` under `"key_failures"`.

## Kumo surfacing

`check_ingest_orphan_skips` (`kumo/data/generic/diskgraph_staging.py`) runs
at every consumer of a TT extension: after the train stage ingest
(`ensure_stage_ingested`), after the predict stage ingest
(`materialize_prediction_task_dataset`), **and — from the persisted
manifest — whenever a present-and-complete extension is reused without
re-ingest** (a later job on the same graph+tt_id, or a retry after a
hard-fail, must not silently bypass the check). Skipped =
`parse + lookup + membership` failures. PK-key-type mismatches may produce
parse or lookup failures depending on the resolution path; these counters do
not account for every null or unparseable FK-free input.

- **skipped == 0** (the same-graph case, since gen pre-filters orphans
  against the gen graph): silent.
- **skipped > 0**: `logger.warning` with the reference count, ~% of seed
  rows (skips count FK *references* — an LP row carries one per target-list
  element — so the % is an upper-bound signal, clamped at 100), per-FK
  `by_context` breakdown, and key samples.
- **≥50%**: escalated wording — "the train table was almost certainly
  generated against a different graph".
- **`KUMO_TTNG_ORPHAN_FAIL_PCT=<percent>`** (unset by default; `0` = fail on
  any skip): raise `TTNGOrphanSkipError` at/above the threshold, always
  after logging the warning; the train activity layer wraps it verbatim into
  a non-retryable `TrainingDataMaterializationError`.

`_validate_ttng_ingest_binding` (`kumo/diskgraph/service.py`) remains a
path/tt_id **sanity** gate only — its stamp is written at re-ingest time with
the same index it later compares against, so it cannot detect cross-graph
reuse. Do not re-introduce a "graphs match" assumption on top of it;
correctness comes from the ingest-time membership verification above.

## Case behavior

| Case | Behavior |
|---|---|
| training graph == gen graph | extension reused or re-ingested with 0 skips |
| same schema, rows added/removed | re-ingest; pkmap rebuilt; FKs re-resolved + verified; orphans skipped + counted + warned |
| referenced parent table gone | pkmap can't build → loud hard error |
| custom TT | identical path (custom is "gen with a source") |
| PK key-type mismatch | May produce parse or lookup failures; counted skips drive warning/fail thresholds. Null or unparseable FK-free inputs can be skipped without counters. |
| reused already-published extension | counters re-checked from the persisted manifest — same warn/fail |
