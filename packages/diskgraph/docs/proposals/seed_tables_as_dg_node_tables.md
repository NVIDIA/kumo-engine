# Seed Tables as DG Node Tables — Plan

A **training seed table** or **prediction seed table** produced from TT-NG output is ingested as a DG node-table-shaped **extension artifact** under `parent_index_path/train_tables/<tt_id>/` or `parent_index_path/pred_tables/<job_id>/`. The resulting persisted seed table is a **runtime-ready seed table**: every stored row is immediately sampleable, is not part of graph topology, has exactly one structural entry link via `ENTITY1` into the parent graph snapshot, and carries task-local supervision / features on the row itself. The parent graph snapshot is untouched. A merged reader (`open_with_tt`) unifies parent + exactly one extension; the sampler walks `seed.edges.bin` at hop 0 and `parent.edges.bin` at hops 1+, emitting a runtime batch dict consumed by PR-28765's `runtime_batch_to_hetero` adapter.

**Phases 1–3 (DG-side) shipped on `vedaant/seed-tables-on-269`.** Phase 4 closes the kumo-side gap so `batch = next(dataloader)` works end-to-end for single-entity / single-target node prediction — both **training** (training seed table) and **inference** (prediction seed table, which normally omits `TARGET1`). Seed-row features are part of the v1 contract when present. Phases 5–9 layer task-type discrimination, LP, and forecasting; each task-type phase ships training + inference together using the same DG infrastructure. Phase 8 deletes the legacy bridge once every task type is on DG-native. Phase 10 (far future) is the kumo-side multi-entity / multi-target consumption unlock per the [ML team's CustomTrainTableSpec](https://www.notion.so/Custom-training-table-335cc5e93e3880c28c24c923c99a0bd9).

**Reading guide:**
- New engineer picking up **Phase 4 work**: §1 architecture → §2 current status → §4 Phase 4 → §8 acceptance for Phase 4.
- Planning **Phase 6+ work**: §1 → §4 sequencing → relevant phase section.
- Reviewing the **shipped Phase 1–3 code**: §2 sub-phase commit table → §9 test specifications.

---

## 1. Architecture

### Design rationale: seed table uses DG node-table storage

A reasonable alternative was: skip materializing seed edges; have the sampler do a runtime pkmap probe from seed PK to parent row id at hop 0. We chose **seed-as-DG-node-table** instead:

1. **Ingest pipeline reuse** — col_stats, sort_keys, multithreaded shards, atomic publish all engage with seed tables identically to entity tables. No degenerate FK-less code path.
2. **Sampler uniformity** — hop 0 walks `edges.bin` like every subsequent hop. Temporal filter via sort_keys applies uniformly.
3. **Storage symmetry** — seed tables look identical to entity tables on disk. Tooling, validators, debuggers, merged reader, bundle producer see one shape.
4. **Bounded future propagation cost** — if seed-table features ever need to participate in message passing, it's a small sampler change to emit the seed namespace in `graphs.main.node_ids` and its outgoing edges in `edge_index_dict`. The runtime-probe alternative would have required ingest changes + data migration (~300 LOC) to add propagation later.

`LinkArray` (the supported LP mode under `ALLOW_ARRAY_TARGETS=true`) is unaffected: `TARGET` carries task-local supervision (a list of RHS values). In v1 (PR270), only `ENTITY1` is materialized as a structural FK in `seed.edges.bin`; `TARGET1` RHS values are not yet materialized as structural FK edges, but this is a v1 scope decision, not an architectural exclusion.

### Storage shape

```
parent_index_path/
├── (parent entity tables, untouched)
├── train_tables/<tt_id>/                    # one subdir per training seed table
│   ├── train_seeds/  val_seeds/  test_seeds/    # per-stage DG node tables
│   │   ├── rowpack/                              # ENTITY1..ENTITYn, TARGET1..TARGETn, [TIMESTAMP], [FEAT*]
│   │   ├── edges.bin                             # one forward FK ENTITY1 → parent.entity
│   │   ├── col_stats.json
│   │   └── (sort keys on TIMESTAMP when present)
│   ├── split.bin                                 # written by tt-gen
│   ├── weight.bin                                # optional sidecar — present iff plan.weight_col set
│   └── manifest.json
└── pred_tables/<job_id>/                    # one subdir per prediction seed table
    └── pred_seeds/                                # same structural/task-local shape, but normally without TARGET1
        ├── rowpack/                              # ENTITY1..ENTITYn, [TIMESTAMP], [FEAT*]
        ├── edges.bin                             # one forward FK ENTITY1 → parent.entity
        ├── col_stats.json
        └── (sort keys on TIMESTAMP when present)
```

Prediction seed tables ship per task-type phase (Phase 4 for node, Phase 7 for LP, Phase 9 for forecasting). Same `ingest_with_stats_from_source` call shape as training seed tables; same merged-reader logic walks both `train_tables/*` and `pred_tables/*`; same sampler emits a runtime batch dict (with `task.target` absent at predict time). One subdir per prediction job, DG's internal sharding handles scale.

**Rowpack columns** come from TT-NG's `ChannelBatchWriter` (`diskgraph-train-table::ttgen_output.rs`):

| Column | Multiplicity | Notes |
|---|---|---|
| `ENTITY1..ENTITYn` | `entity_dtypes.len()` — supported by writer for all OutputModes. | n ≥ 1. Parent entity foreign keys. |
| `TARGET1..TARGETn` | `target_dtypes.len()` on Node / LinkExploded / LinkArray / Forecasting (Phase 3.K vec-ified Link + Forecasting; Node was already vec'd). PassThrough has no TARGETs. | Present on training seed tables as supervision. Prediction seed tables normally omit TARGET entirely. |
| `TIMESTAMP` | Optional. Present only for temporal tasks. | Sort-key column when present. |
| `FEAT1..FEATn / LAG_FEAT*` | `feat_columns: Vec<ExtraColumnOutput>` (Phase 3.K). | Supported in v1 as task-local seed-row features when present. Scalar or array-typed per slot. |
| `split_projection` / `ENTITY_INCLUDE_BIT` | Optional. Existing extra-column slot, reserved for split projection. | |

**Additional artifacts (not rowpack columns):**
- `split.bin` — per-row stage byte (train/val/test). Mechanism documented in `tt-ng-architecture-and-migration.md`.
- `weight.bin` — present iff `plan.weight_col` is set. DG sampler auto-detects via `reader.tt_extension_root()` (Phase 3.J). Per-row fixed-stride f32.
- `arange.bin` — transitional compatibility sidecar when present in TT-NG artifacts. It is **not** part of the DG-native seed-table semantics.

**Edges:** one scalar, one-directional forward FK from `ENTITY1` to the parent's entity (LHS) table — materialized in `seed.edges.bin`. Every persisted seed row has exactly one such structural entry link. No reverse edges. No RHS edges in v1 — for LP, `TARGET1` carries RHS values as task-local supervision lists; materializing RHS values as structural FK edges (a parallel `seed.edges.bin` for the target side) is not yet implemented in PR270 but is not architecturally excluded.

### Reader and sampler

`GraphIndexReader::open_with_tt(parent_root, tt_id)` merges parent + the one chosen extension into a unified catalog. Runtime opens parent + exactly one seed table per run (training or inference); there is no multi-seed-at-once case. `discover_tts(parent_root)` enumerates available tts for admin/listing.

The sampler walks `seed.edges.bin` at hop 0 (returning entity row ids — no PK lookup; resolution happened at ingest) and `parent.edges.bin` at hops 1+. The seed namespace is **never** placed in `graphs.main.node_ids` or `edge_index_dict` — the adapter consumes it from `task.*` as a separate task bag at `hetero[train_table_name]`. Persisted row identity at runtime is the **persisted seed-row id** assigned after ingest-time dropping / filtering, not the raw TT input row position.

### Three layers of multi-* support

The DG writer, DG sampler, and kumo adapter each support a different subset today:

| Capability | DG writer | DG sampler `sample_runtime_batch` | Kumo adapter (PR-28765) |
|---|---|---|---|
| Multi-entity (`ENTITY1..ENTITYn`) | ✅ All OutputModes | ❌ One cross-boundary FK; errors loudly on `n > 1` | ❌ Single seed bag only |
| Multi-target (`TARGET1..TARGETn`) | ✅ Node + Link + Forecasting (Phase 3.K) | ✅ Routed by column-name prefix (Phase 3.J) | ❌ Single Array per adapter docstring |
| Feat columns (`FEAT*/LAG_FEAT*`) | ✅ Writer slot (Phase 3.K) | ✅ Routed to `raw_features[<seed_namespace>]` by prefix (Phase 3.J) | ⚠️ Slot wired; tf-lowering not exercised end-to-end |
| Weight | ✅ Sidecar `weight.bin` | ✅ Auto-detected via `tt_extension_root()` (Phase 3.J) | ✅ Consumed from `task.weight` |

Phase 4 ships the **single-entity / single-target** subset (matching current kumo) plus seed-row feature support when present. Phase 10 unlocks kumo-side multi-entity / multi-target consumption.

### Temporal filter scope

When the seed has a `TIMESTAMP` column:
- **Hop 0** — seed's TIMESTAMP is the batch anchor; the edge trivially passes.
- **Hop 1** — current implementation applies `TemporalMostRecent` on parent edge sort key; only neighbors with `dst_time ≤ anchor` are emitted.
- **Hop 2+** — **semantic scope still open.** Current implementation does not enforce all-hop visibility, but this proposal does **not** treat hop-1-only filtering as a settled domain contract.

When the seed has no `TIMESTAMP` (non-temporal task): uniform fanout at all hops, no filter.

### LinkPredTrainTable — replaced on-disk in Phase 6

`LinkPredTrainTable` (`kumoml/enterprise/data/link_pred_train_table.py:56-280`) is an in-memory CSR/COO index over the full training set's `(LHS row × (time, RHS))` fkey tensor. Two use cases (`base_link_module.py:258, 262, 306, 319`):

1. **Per-row label fetch** (`query`, `query_stage`) — given (lhs, t), return positive RHS values. Redundant with the seed rowpack's `TARGET1` List column.
2. **Cross-row past-visit exclusion** (`query_until`, `query_stage_until`) — given (lhs, t), return positives across the training set where t' < t. Used to exclude past positives when sampling negatives. A range query over the full training history, not per-seed data.

**Phase 6 replaces both on-disk.** Past-visit exclusion is structurally a time-range query on parent's LP-source edges — DG already supports this via `TemporalMostRecent` sort keys. Phase 6 sub-step #6 (collision check) lands it. Stage-filtered exclusion (`query_stage_until`) needs a Phase 6 schema decision: per-edge stage column, per-stage edge bundles, or stage bitmap sidecar.

**Why this matters at scale.** The in-memory index scales with total positives — billions on production LP datasets. Keeping it in memory violates the "no Python artifacts proportional to seed count" invariant the rest of this plan upholds. Phase 6's on-disk path makes LP-at-scale viable. A bounded eval-only cache (val/test positives, not full train) may remain if eval-step latency requires it.

### Internal unification note

The same DG implementation machinery (`diskgraph-train-table/`, `OutputMode`, `ChannelBatchWriter`) produces both training seed tables and prediction seed tables. That implementation sharing does **not** create a new domain object. This proposal uses **training seed table** / **prediction seed table** terminology throughout, and reserves `target` for labels / target expressions / `TARGET1`.

---

## 2. Current status

**Branch:** `vedaant/seed-tables-on-269` (`/home/vedaant/kumo-diskgraph-impl`). **HEAD** `b19a935a` (2026-05-15).

**Phases 1, 2, 3 complete.** End-to-end pipeline working: TT-NG ingested → merged reader opens parent + sub-index → `engine.sample_runtime_batch(...)` produces a RuntimeBatch matching PR-28765's adapter contract → pybind `RuntimeBatchIterator` exposes a worker-sharded iterator over PyDict batches.

**Phase 1 (Ingest plumbing).** FKeySpec.external_parent + bidirectional + validate relaxation, external PkHashStore loader, IngestOptions.parent_index_path, edge-phase reverse-edge guards, coordinated-layout write side (`with_external_node_types`). Cross-boundary FK wired into `execute_tt_plan` via `ingest_stage_parquets_into_extension` (commit `3be9facd`). 19+ tests.

**Phase 2 (Merged reader).** `open_with_tt` + `discover_tts` + `merged_edge_relations()` namespaced extension src_types + `DiskGraphEngine::from_reader` + EdgeUnpacker dst_bits=64 guard + coordinated-layout read side (`open_with_external_node_types`) + `tt_extension_root` for sidecar discovery. 7+ tests.

**Phase 3 (Sampler).** 11 sub-phases A–K, one commit each:

| Sub | Commit | What |
|---|---|---|
| 3.A | `4313c4bd` | `RuntimeBatchOutput` collator. Synthesizes dict matching adapter contract. Seed-src exclude is non-emission. |
| 3.B | `b3a0bd79` | `engine.sample_runtime_batch(reader, stage_table, batch_row_ids, ...) -> RuntimeBatch`. Single entry point. E2E test pinned. |
| 3.C | `c5014344` | Per-table `local_to_merged_rel: Vec<Vec<u16>>` translation. Fixes local/merged rel-id collision. Hop-1 temporal-leak test pinned. |
| 3.D | `d5b64c0c` | Pybind `RuntimeBatchIterator` replaces PR-269 dual-engine iterator. `discover_tts` pyfunction. Python wrapper `make_runtime_batch_iterator`. |
| 3.E | `3c786db4` | Stop hardcoding `fused`/`swap_edge_direction`. Caller-controlled kwargs. Non-fused path post-fetches features for parity. |
| 3.F | `2ae4c563` | Acceptance test (`test_runtime_batch_iterator_e2e.py`). 6 assertions: dict contract, multi-worker sharding, shuffle determinism, drop_last, fused/non-fused parity. Plus perf bench. |
| 3.G | `2dd2551f` | Production-path e2e via `execute_tt_plan`. Manifest-shape ownership resolved by PR-28765's `_flat_manifest_writer.py`. |
| 3.H | `be13980c` | Coordinated layout fix. ~8× per-edge size savings on cross-boundary edges. |
| 3.I | `06a9894a` | DG-side fused-mode heuristic. `engine::choose_fused_mode` at iterator construction; `fused=None` default. |
| 3.J | `e22ccc02` | Sampler projection (TARGET-prefix → `task.target`; non-task cols → `raw_features[<seed_namespace>]`) + weight.bin sidecar auto-detect. |
| 3.K | `b19a935a` | tt-ng schema additions: multi-target Link/Forecasting, `feat_columns: Vec<ExtraColumnOutput>` writer slot, `push_feat_values` API. Hot-path alloc-free. |

**Perf baseline** (commit `2ae4c563`): 10K users × 8 orders/user (80K orders) + 10K seeds, batch_size=128, single worker — non-fused **500 batches/s** / fused **575 batches/s** (~15% faster, cache locality), 162 MB peak RSS. Caveat: small fixture, single worker. 100M-seed × 8-worker validation is a deferred perf follow-up.

**Test scoreboard** (all green at `b19a935a`): `cargo test --workspace` 1509 pass / 0 fail · `cargo clippy --workspace --all-targets -- -D warnings` clean · `pytest python/tests/` 92 passed / 2 skipped / 0 failed.

**What's next: Phase 4.** Closes the kumo-side gap to reach `batch = next(dataloader)`. PR-28765 ships ~95% of the kumo machinery; remaining work is ~100-120 LOC: iterator rename, target shape alignment, E2E smoke test, and confirming `train_table_name` ↔ `<seed_namespace>` matches end-to-end between PR-28765's adapter and our sampler emission. See §4 Phase 4.

---

## 3. Environment variables, flags, and branch strategy

### Environment variables

| Var | Effect |
|---|---|
| `KUMO_DISKGRAPH_TT_NG_ENABLED` | Gates the DG-native loader path. Off → legacy bridge. On → DG-native. Read via `_tt_ng_dispatch.use_tt_ng()`. |
| `ALLOW_ARRAY_TARGETS` | Controls LinkArray vs LinkExploded. We target LinkArray (=true) as the supported LP mode. LinkExploded (=false, legacy HStack) is not maintained under DG-native. |
| `DG_FUSED_L2_TARGET` | Override the L2-sized threshold (default 512 KB) used by `engine::choose_fused_mode` to pick fused vs non-fused sampling. |

### Config flags (per-FK, not env)

| Flag | Effect |
|---|---|
| `FKeySpec.external_parent` | `true` → dst_table lives in parent index; ingest mmaps external pkmap. |
| `FKeySpec.bidirectional` | `false` → skip reverse_batcher; one-directional edges only. |

### Branch strategy

- Base branch: PR 269 (`ben/tt-into-index-ingest-rebased`) — the TT ingest foundation. PR 269 targets `tt-ng`.
- Our work lives on `vedaant/seed-tables-on-269` layered on PR 269. Phases 1–3 changes layer cleanly on top: cross-boundary edges in Phase 1 ingest, merged-reader single-engine replacing PR 269's two-engine `task_batch_iterator`, and the new `RuntimeBatchOutputHandler`.
- Eventually merges back into `tt-ng` along with PR 269.
- Phase 4 (kumo Python loader) targets a kumo branch that depends on PR-28765 + Phases 1–3 PRs.
- `main` branch of `kumo-diskgraph` remains untouched until `tt-ng` itself merges to main.

---

## 4. Phased plan

The sequence groups by task type. Each task-type phase ships both **training** and **inference** (prediction seed tables) using the same DG infrastructure — prediction seed tables normally omit `TARGET1`, are ingested at `pred_tables/<job_id>/`, and otherwise share the same extension mechanics. In this document, `TARGET1..TARGETn` names the physical rowpack columns, while `task.target` / "target" means the logical supervised payload projected out of those columns for the runtime batch. Phases 1–4 deliver node prediction end-to-end. Phase 5 forks the dispatch so LP and forecasting stay on legacy. Phases 6–7 ship LP, Phase 9 ships forecasting (9a PQ + 9b custom-TT). Phase 8 deletes the legacy code once every task type is on DG-native. Phase 10 is the far-future kumo-side multi-entity / multi-target expansion, not a rollback of v1 seed-row feature support.

| Phase | Scope | Task types | DG / kumo |
|---|---|---|---|
| **1** ✅ | Ingest plumbing | All | DG |
| **2** ✅ | Merged reader | All | DG |
| **3** ✅ | Sampler (11 sub-phases A–K) | All schemas | DG |
| **4** | Node prediction training + inference (PR-28765 + narrow deltas + pred-ingest entry point) | Node prediction | kumo |
| **5** | Task-type discriminator at dispatch | Node prediction (initial) | kumo |
| **6** | LP sampler primitives | LP | DG |
| **7** | LP training + inference (adapter LP arm + dispatch + pred wiring) | LP | kumo |
| **9** | Forecasting training + inference (9a PQ + 9b custom-TT) | Forecasting | DG + kumo |
| **8a** | Legacy deletion sweep, gated behind `not use_tt_ng()` | All | kumo |
| **8b** | Flip default to ON + remove env-off branches | All | kumo |
| **10** *(far future)* | Kumo-side multi-entity / multi-target consumption (broader CustomTrainTableSpec) | All | kumo (+ small DG-side) |

### Phase 1 — Ingest plumbing ✅ SHIPPED

**Problem:** seed tables need to write cross-boundary FK edges from `ENTITY1` to the parent entity table, but DG ingest's edge phase assumes the destination table is in the same `GraphConfig`. We need an external-pkmap path.

**What shipped:**
- `load_external_pk_hash_shard(parent_index_path, dst_table)` — reads `pk_hash-*.bin` + `pk_data-*.bin` from `parent_index_path/nodes/<dst_table>/` and constructs a shard the existing edge phase probes via `pk_hash_store.shard(&dst_table, dst_shard)`. Validates shard count, PK type compatibility, format version. Closes PR-269's `tt_executor.rs:259` TODO.
- `FKeySpec.external_parent` + `FKeySpec.bidirectional` flags. Defaults preserve existing behavior.
- `parent_index_path: Option<&Path>` on `ingest_with_stats_from_source`.
- Validate relaxation: skip in-config `dst_table` check when `external_parent: true`; verify `src_column` exists.
- Edge-phase guards in `lib.rs`: `reverse_batcher.push` / `flush` / count-insert gated on `bidirectional`. Reverse `RelationDef` + `DestFkeyBatcher.push` skipped for external FKs.
- **Coordinated layout (write side)** — `GraphIndexBuilder::with_external_node_types(parent.node_types)` injects parent's `node_types` into the in-memory layout-computation scope. Cross-boundary FKs encode `dst_id` with `bits_for_id(parent.users.max_id)` instead of falling back to `dst_bits=64`. **~8× per-edge size savings** at production scale. Persisted `index_metadata.json` unaffected.

**Acceptance:** 19+ tests including `test_seed_edges_dst_bits_use_parent_max_id`. All existing `cargo test -p diskgraph-ingest` continues to pass.

### Phase 2 — Merged reader ✅ SHIPPED

**Problem:** runtime opens parent + one extension. The reader needs to merge catalogs, namespace extension tables, resolve cross-boundary FKs, and expose sidecar paths to the sampler.

**What shipped:**
- `discover_tts(parent_root) -> Vec<String>` — finalized tts only, sorted.
- `GraphIndexReader::open_with_tt(parent_root, tt_id) -> Self` — parent + exactly one extension, namespaced under `<tt_id>/<bare>`. Populates `tt_extension_root: Option<PathBuf>` for sampler-time sidecar discovery.
- Merged `col_stats` — walk extension `col_stats.json` files, merge into an in-memory dict at open time.
- Namespaced extension catalog keys (`<tt_id>/<table_name>`); `merged_edge_relations()` exposes parent and extension src_types in one unified relation list.
- Cross-boundary FK resolution via Phase 1's `PkHashStore` infrastructure (reuse, don't duplicate).
- **Coordinated layout (read side)** — `merge_extension_into` calls `open_with_external_node_types(parent.node_types)`. Paired with Phase 1's write-side fix; read and write agree on `entry_bytes`.
- `EdgeUnpacker dst_bits=64` guard — fail loudly if a segment's `dst_bits` would force the legacy 64-bit fallback.
- `DiskGraphEngine::from_reader` — new constructor that takes the merged reader directly, eliminating PR 269's two-engine `task_batch_iterator` pattern.

**Staleness contract:** `open_with_tt` constructs `Arc<ShardReader>`s once per open; snapshotted state for the open's lifetime. Re-open against an advanced parent fails loudly when the seed-table extension no longer matches the parent graph snapshot it was built against. Orchestrator owns advance detection; reader owns the snapshot. The long-term explicit `parent_snapshot_id` mechanism is tracked separately in kumo [#28849](https://github.com/kumo-ai/kumo/issues/28849); do not treat path names or `_MANIFEST.json` checksums as the desired semantic identity contract.

**Acceptance:** 7+ tests. Existing `GraphIndexReader::open` regression-tested.

### Phase 3 — Sampler + RuntimeBatchOutputHandler ✅ SHIPPED

**Problem:** the sampler walks `seed.edges.bin` at hop 0, applies the temporal filter at hop 1, and emits a dict matching PR-28765's adapter contract. The seed never enters graph topology; it lives only in `task.*`.

**What shipped:** 11 sub-phases A–K (see §2 Phase 3 table for commits). Notable design decisions:

- **Single entry point**: `engine.sample_runtime_batch(reader, stage_table, batch_row_ids, ...) -> RuntimeBatch`. Walks `seed.edges.bin` for resolved parent ids (no PK lookup — Phase 1 ingest already resolved them).
- **Seed-src exclude is non-emission**, not a filter. The seed namespace is simply never added to `graphs.main.node_ids` or `edge_index_dict`. Adapter consumes it from `task.*` as a separate task bag.
- **Per-table `local_to_merged_rel: Vec<Vec<u16>>` translation** — fixes a latent local/merged rel-id collision where extension shards' local rel ids collide with parent's at sample time.
- **DG-side fused vs non-fused heuristic** — `engine::choose_fused_mode(reader, plan, batch_size)` computes the expected feature working set and picks fused (cache locality) below L2 / non-fused (parallel map-reduce) above. `RuntimeBatchIterator(fused=None)` invokes the heuristic. Explicit `Some(true)/Some(false)` overrides. Decision lives in DG so it has an owner after Phase 8 deletes kumo's sampler abstraction.
- **Column-name-prefix routing for task fields**: `TARGET*` → `task.target` (multi-column RecordBatch); `ENTITY*` / `TIMESTAMP` → filtered out (already in `task.tt_stage_node_ids` / `task.seed_timestamps`); everything else → `raw_features[<seed_namespace>]`. Weight auto-loaded from sidecar via `reader.tt_extension_root().map(|root| root.join("weight.bin"))`. Python iterator doesn't pass paths.
- **Multi-target / feat_columns schema** (Phase 3.K) — `OutputMode::{LinkExploded, LinkArray, Forecasting}` accept `target_dtypes: Vec<DataType>`; new `feat_columns: Vec<ExtraColumnOutput>` writer slot. Hot path stays alloc-free via `(Option<&Value>, &[Value])` and `(Vec<Value>, &[Vec<Value>])` signatures. Sampler picks up new columns automatically via 3.J's prefix routing.

**Acceptance:** runtime batch dict matches adapter contract on the fixture; hop-1 temporal filter applies; seed namespace never appears in topology; parent index bytes unchanged after seed-table ingest.

### Phase 4 — Node prediction end-to-end (training + inference)

**Goal:** `batch = next(dataloader)` works end-to-end for single-entity, single-target node prediction, for both **training** (training seed table → trainer step) and **inference** (prediction seed table → predict step). Multi-entity / multi-target consumption is deferred to Phase 10. Seed-row features are in scope for v1 and should lower through the adapter when present. This phase is parity with current kumo on entity/target shape, not a rollback of already-supported seed-row feature semantics.

**What "multi-entity node prediction" would mean here:** one seed row would carry multiple structural entity slots such as `(ENTITY1=user_id, ENTITY2=device_id, TIMESTAMP, TARGET1)` for a single supervised example. TT-NG's writer can encode that schema, but current kumo node-prediction consumption is a single-seed-bag / single-entry-point shape, so DG-native v1 rejects multi-entity node prediction explicitly rather than silently falling back.

**State of PR-28765 (`ben/dg-tt-ng-trainer`):** open draft, +4143 LOC across 17 files. Ships ~95% of the kumo-side machinery:

| Component | File | LOC | Status |
|---|---|---|---|
| Adapter (dict → HeteroData) | `kumo/data/generic/diskgraph_runtime_adapter.py` | +497 | ✅ Consumes our exact `task.{...}` contract. `train_table_name` parameter is the namespace knob. |
| Lightning datamodule | `kumo/train/datamodule/diskgraph_lightning_data.py` | +236 | ⚠️ Imports `make_task_batch_iterator` (old name) — needs rename. |
| DG trainer subclass | `kumo/train/diskgraph_trainer.py` | +413 | ✅ |
| Nested trainer subclass | `kumo/train/diskgraph_nested_trainer.py` | +310 | ✅ |
| Manifest writer | `kumo/engine/diskgraph/_flat_manifest_writer.py` | +133 | ✅ Owns the iterator-compatible manifest shape; translates from `execute_tt_plan`'s output during materialize. |
| Autotrainer dispatch hook | `kumo/automl/auto_trainer.py` | +109 | ✅ Routes via `JobConfig.diskgraph_task_dataset_bundle_dir`. |
| Materialize-activity tt-ng branch | `materialize_training_task.py` | +24 | ✅ Gates on `use_tt_ng()`. |
| Unit tests | 6 files | +2371 | ✅ Adapter, lightning data, trainers, manifest writer, dispatch, materialize. |

**Remaining work (~100-120 LOC):**

1. **Iterator rename** in `diskgraph_lightning_data.py:124`: `make_task_batch_iterator` → `make_runtime_batch_iterator`. Verify kwarg signature matches (`manifest_path`, `stage`, `batch_size`, `shuffle`, `num_workers`, `epoch_seed`, `drop_last`, plus `fused=None` and `swap_edge_direction` defaults).
2. **Target shape: Array vs RecordBatch.** Adapter docstring says `pyarrow.Array` (singular); we emit a multi-column RecordBatch. For parity: either adapter does `.column(0)` when N=1, or DG emits a singular Array when N=1. Pick one end and adapt. ~10 LOC + 1 round-trip test.
3. **`train_table_name` ↔ `<seed_namespace>` string alignment.** PR-28765's adapter takes `train_table_name` as a parameter and uses it to (a) place the seed bag at `hetero[train_table_name]` and (b) look up `raw_features[<train_table_name>]` when lowering features to `hetero[train_table_name].tf`. Our sampler emits `raw_features["<seed_namespace>"]`. These strings must be the same value end-to-end, or feature lowering won't find the seed features. Whoever wires Phase 4 picks one canonical source (e.g., the kumo-side manifest writer sets both) and ensures the adapter is invoked with that value.
4. **E2E smoke test in kumo-tests** (~80 LOC). PR-28765's tests are unit-level. The full pipeline — ingest → `execute_tt_plan` → materialize → `DiskGraphNestedTrainer.fit()` — has no smoke test yet. Build a small node-prediction fixture, run one training step under `KUMO_DISKGRAPH_TT_NG_ENABLED=1`, assert loss is finite and no errors.

**Inference (prediction seed tables).** Prediction seed tables share the same structural/task-local layout as training seed tables, but `TARGET1` is normally absent by design. They live at `parent_index_path/pred_tables/<job_id>/` (parallel to `train_tables/<tt_id>/`). The reader, sampler, and adapter machinery built for training all work for inference unchanged; the adapter consumes a HeteroData bag without `.y`. Phase 4 ships this for node prediction:

1. **New pred-ingest entry point in `kumo/engine/diskgraph/`** — given a pred parquet path and a parent index path, ingest as a DG node table at `parent_index_path/pred_tables/<job_id>/` using the same `ingest_with_stats_from_source` call shape as TT seeds (one external one-directional FK on `ENTITY`). Replaces the legacy `prediction_artifacts.py` pipeline. Cross-cutting — used by all task types in their respective phases. ~80 LOC.
2. **Merged reader walks `pred_tables/*`** in addition to `train_tables/*`. Same `_SUCCESS`-gated discovery, same catalog merging with `<job_id>/pred_seeds` namespacing. ~30 LOC reader extension.
3. **`materialize_prediction_task.py` node-prediction arm** rewritten to call the new ingest entry point. `JobConfig.diskgraph_task_dataset_bundle_dir` set to `pred_tables/<job_id>/`. ~80 LOC.
4. **Adapter handles missing `.y`** — PR-28765's adapter is likely fine here (the `target` field becomes empty/None for pred), but Phase 4 adds an explicit test (`test_pred_table_no_target_column_in_schema`).
5. **Layout: one subdir per pred job, internal DG sharding.** DG's `build_temp_shards` rayon pool handles parallelism (informed by PK distribution + memory budget). Crash recovery via DG's `resume_from_phase`.

Legacy pred artifact deletions (`build_prediction_table_edges_artifact`, `_diskgraph_partition_table`, fkey_cols/mask/reverse-edge sidecars) go in Phase 8a's deletion sweep.

**Acceptance:** PR-28765 lands. `KUMO_DISKGRAPH_TT_NG_ENABLED=1` + a node-prediction training seed table runs **one training step** end-to-end, and a node-prediction prediction seed table runs **one prediction step** end-to-end. Adapter consumes both dict shapes without errors, including seed-row features when present. Existing kumo tests (env-off path) unchanged.

### Phase 5 — Task-type discriminator at dispatch

**Problem:** `use_tt_ng()` is currently a single boolean. When on, ALL TTs (node + LP + forecasting) route through TT-NG. For fork-not-delete, only node-prediction TTs should take the DG-native path; LP and forecasting must stay on legacy until their respective phases ship.

**Why fork instead of delete:** the legacy 3-tuple datamodule and CSC adapter are shared between node prediction and LP. `TwoStageNodeGNNLightningModule.train_step` unpacks the 3-tuple regardless of inner module type. `_build_fkey_seeds` / `_build_fact_feats` / `_attach_train_table` feed both node and LP modules. Deleting them in Phase 5 would break LP (still on legacy through Phase 6–7).

**Targets (additive, no deletion):**
- Add `is_node_prediction_tt(plan_or_manifest)` helper to `_tt_ng_dispatch.py` — discriminates by TT type (looks at `PredictionType` in the resolved plan or `task_type` field in the materialize manifest).
- Compose `use_tt_ng_for_task(plan_or_manifest)` as `use_tt_ng() AND is_node_prediction_tt(...)`. Update the two call sites (`materialize_training_task.py:437`, `auto_trainer.py` dispatch).
- LP and forecasting TTs continue through the legacy 3-tuple path; `_build_fkey_seeds`, `_build_fact_feats`, `_attach_train_table`, CSC adapter all stay live.
- Add instrumentation/metrics (counter or log line) so production can confirm node-prediction TTs are taking the new path.

**Order (TDD):**
1. Write fork tests: env-on + node-TT → DG path; env-on + LP-TT → legacy; env-on + forecasting-TT → legacy; env-off + any-TT → legacy.
2. Add `is_node_prediction_tt` + `use_tt_ng_for_task` helpers.
3. Update the two call sites.
4. Verify legacy behavior bit-identical when discriminator returns False (regression test).

**LOC:** ~30 LOC helpers + ~20 LOC call-site updates + ~50 LOC fork tests ≈ 100 LOC.

**Acceptance:** all four dispatch cases produce the right path. Existing tests in both repos continue to pass.

### Phase 6 — LP sampler primitives

**Problem:** the DG sampler today doesn't expand `TARGET1` lists, doesn't do streaming negative sampling, and doesn't time-filter the Uniform mode. LP needs all three.

**Schema already available (Phase 3.K):** `OutputMode::{LinkExploded, LinkArray}` are vec-shaped. tt-ng can already emit single-target or multi-target LP TT schemas. Phase 6 scopes to single-target LP (matching current kumo); multi-target LP consumption is Phase 10.

**Order (TDD):**
1. Write tests first (failing): TARGET1 list expansion correctness, `sample_from_edges` shape, time-filtered Uniform anchor logic, negative-sampler distribution, collision check, hand-checked subgraph fixture.
2. `engine.expand_link_targets(reader, seed_row_ids) -> Vec<Vec<u64>>` — per seed, read `TARGET1` list column, probe parent.rhs.pkmap for each element (using Phase 1's shared `PkHashStore`), return per-seed RHS row id lists. Reused by Standard and ShallowRHS.
3. `engine.sample_from_edges(reader, seed_lhs, seed_rhs, seed_neg, fanout_plan) -> RuntimeBatch` — joint sampling entry point. Walks the fanout from each frontier; emits LP-shaped `RuntimeBatch`.
4. **Time-filtered Uniform sampling mode.** Today's Uniform path ignores time; add the anchor logic from `TemporalMostRecent` so Uniform-with-anchor is selectable for LP plans that need it.
5. `engine.sample_with_negatives(reader, seed_row_ids, strategy: NegStrategy, n_per_seed) -> Vec<Vec<u64>>` — streaming negative sampler. Strategy enum: `Uniform`, `Popularity`.
6. **On-disk positive-collision check** via parent's edge index. For each candidate negative, probe parent edges to verify it's not a true positive for (lhs, anchor_time). **This replaces `LinkPredTrainTable.query_until`.**
7. **Stage-filtered exclusion design step.** `query_stage_until` (excludes val/test positives during train negative sampling) needs a schema decision: per-edge `stage` column, per-stage edge bundles, or stage bitmap sidecar. Pick one in this phase.
8. Performance benchmarks against legacy GE LP on the same fixture (target: within 1.5× legacy on representative workload).

**Entry points (for Phase 7 to consume via pybind):** `expand_link_targets`, `sample_from_edges`, `sample_with_negatives`, `NegStrategy` enum.

**LOC estimate:**
- `expand_link_targets` + pkmap probe reuse: ~150 LOC + ~120 LOC tests.
- `sample_from_edges` entry + per-frontier walking: ~800 LOC + ~350 LOC tests.
- Time-filtered Uniform: ~120 LOC + ~80 LOC tests.
- `sample_with_negatives` + strategy enum: ~400 LOC + ~200 LOC tests.
- Collision check (parent edge probe): ~200 LOC + ~120 LOC tests.
- Pybind exposure: ~100 LOC + ~80 LOC tests.
- **Total: ~1770 LOC implementation + ~950 LOC tests ≈ 2720 LOC.**

**Acceptance:** LP sampler primitives produce correct subgraphs on a hand-checked fixture. Negative sampler distribution within statistical tolerance of declared strategy. Collision check + stage-filtered exclusion excludes all true positives across stages. Performance within 1.5× legacy GE LP on representative workload.

### Phase 7 — LP end-to-end (training + inference)

**Problem:** PR-28765's adapter is shaped for node prediction (single seed bag, single `task.target` column → `y` tensor). LP HeteroData is different: LHS + RHS bags (Standard) or single LHS bag with TARGET as list of RHS row ids (ShallowRHS). The adapter needs LP-shaped code paths added — not a rewrite, but a new dispatch arm. Same machinery serves LP **training** and LP **inference** (MIPS index built once via the rhs dataloader).

**Order (TDD):**
1. Write LP-adapter tests: ShallowRHS shape (single LHS bag + RHS list), Standard shape (LHS + RHS + negatives bags). Pin against hand-checked HeteroData snapshots.
2. **Extend adapter** (`diskgraph_runtime_adapter.py`) with `runtime_batch_to_hetero_lp(...)` (or a `task_type`-dispatched arm of the existing function): consumes Phase 6's LP-shaped `RuntimeBatch` (with `task.{lhs_input_ids, rhs_input_ids, negatives?, target_lists, ...}`), produces appropriate HeteroData.
3. **Pybind / Python entry point** — confirm Phase 6 exposes a callable wrapper (e.g., `make_lp_runtime_batch_iterator(manifest_path, ...)`) usable by the LP datamodule. If Phase 6 reuses `make_runtime_batch_iterator` with an `lp_mode=True` kwarg, document it; if a separate entry, name it explicitly.
4. **Rewrite LP datamodules** (`lightning_neighbor_link_data.py`, `lhs_node_lightning_data.py` for ShallowRHS) to consume the DG-native LP iterator output instead of PyG's `LinkLoader`.
5. **Wire `LinkGNNLightningModule` and `ShallowRHSLinkGNNLightningModule`** to consume the new HeteroData shape. Model code unchanged — only the datamodule output contract changes.
6. **Extend `is_node_prediction_tt`** (from Phase 5) to also discriminate LP TTs so they route to the new path when Phase 7 ships.
7. **Phase out `LinkPredTrainTable`** as past-visit exclusion source — Phase 6's collision check covers `query_until` / `query_stage_until`. Eval-only `query` / `query_stage` can stay in a bounded in-memory cache (val/test positives only) if eval-step latency requires it.
8. **LP inference (prediction seed tables).** Prediction seed tables for LP use the same `pred_tables/<job_id>/` infrastructure introduced in Phase 4. `materialize_prediction_task.py` LP arm rewritten to call the shared pred-ingest entry point; LP-specific adapter arm handles the no-`.y` shape. Standard + ShallowRHS both work at predict time identically to training (MIPS index built once via `rhs_dataloader`). ~80 LOC.
9. Tests green; metric parity with legacy GE LP on the e2e fixture for both training and predict.

**LOC estimate:**
- Adapter LP path additions (training + inference shapes): ~400 LOC + ~250 LOC tests.
- LP datamodule rewrite (Standard + ShallowRHS): ~350 LOC + ~150 LOC tests.
- LightningModule wiring: ~100 LOC + ~50 LOC tests.
- LP arm of `materialize_prediction_task.py`: ~80 LOC + ~40 LOC tests.
- Phase 5 discriminator extension: ~30 LOC + ~30 LOC tests.
- **Total: ~960 LOC implementation + ~520 LOC tests ≈ 1480 LOC.**

**Acceptance:** both LP modules work end-to-end on DG-native for training AND inference. Metric parity with legacy GE LP on the e2e fixture.

### Phase 9 — Forecasting end-to-end (training + inference)

Forecasting splits into two sub-phases: **9a** (PQ forecasting) lands the sampler + adapter forecasting machinery; **9b** (custom-TT forecasting) builds on it to handle segmented sweep and custom-TT-specific column shapes. The pred-table machinery from Phase 4 carries forecasting inference unchanged; the per-task wiring is small.

#### Phase 9a — PQ forecasting

**Problem:** forecasting tasks emit `autoregression_labels` and (optionally) `year_over_year_labels` in addition to the standard task fields. The sampler needs to route these into dedicated `task.*` slots; the adapter needs a forecasting arm.

**Label computation moves from kumo (online) to DG (offline).** Today kumo computes AR / YoY labels **on the fly at batch-construction time** — datamodule code reads the entity table's historical values per seed and computes the lookback window + YoY offset. Under DG-native, those labels are pre-computed during `execute_tt_plan` (via existing ttgen forecasting ops in `ttgen_executor.rs`) and materialized as rowpack columns on the seed table. The kumo on-the-fly computer becomes dead code on the env-on path and is deleted in Phase 8a.

What this changes for a Phase 9a implementer:
- **Identify the kumo on-the-fly computer.** Likely a `compute_autoregression_labels` / `compute_yoy_labels` function (or hook in the datamodule's `_prepare_batch` path) inside `forecasting_lightning_data.py`. Read the entity table's historical values per seed; compute the AR window + YoY lookback.
- **DG-native replacement.** Labels arrive pre-computed via `task.autoregression_labels` / `task.yoy_labels` slots in the runtime batch dict (Phase 9a sampler step). Adapter places them on `hetero[train_table_name]` as tensor attributes. `forecasting_gnn_module.py` consumes them directly — no online computation in the training step.
- **Parity test (separate from training-metric parity).** Pin AR / YoY label values bit-for-bit (or epsilon-tolerance for floating-point arithmetic) between kumo's on-the-fly computation and DG's pre-computed columns on a fixture seed set. Tighter than diffing training loss — the model is involved in the latter and would smear out small label discrepancies.
- **Phase 8a deletion target:** the on-the-fly label computer in `forecasting_lightning_data.py` (and any caller in `forecasting_gnn_module.py` that branched on "are labels present? else compute them"). Listed under the "Forecasting-specific legacy paths" bullet in §4 Phase 8a.

**Sampler output shape:** AR and YoY flow as dedicated `task.*` slots, NOT extra columns inside `task.target`:

```
task.target                  : RecordBatch (TARGET1..TARGETn, present only if target_dtypes non-empty)
task.autoregression_labels   : pyarrow.ListArray (always present for Forecasting)
task.yoy_labels              : pyarrow.ListArray | None (present iff yoy_dtype.is_some())
```

Rationale: AR and YoY are conceptually distinct labels, not extra target columns. Adapter dispatches cleanly by task_type without inspecting RecordBatch column names. Keeps the `task.target` Array-vs-RecordBatch decision from Phase 4 orthogonal to forecasting.

**Order (TDD):**
1. Write tests pinning all four schema variants on a hand-checked fixture + parity test vs legacy forecasting metric:
   - target_dtypes non-empty + yoy present.
   - target_dtypes non-empty + yoy absent.
   - target_dtypes empty + yoy present.
   - target_dtypes empty + yoy absent.

   For Phase 9a parity scope, `target_dtypes.len() ≤ 1` is the only supported case (matching current kumo). Multi-target forecasting kumo-side consumption is Phase 10.
2. **Sampler extension:** `engine.sample_runtime_batch` for Forecasting plans reads `autoregression_labels` and (if present) `year_over_year_labels` from the seed rowpack into the new `task.*` slots.
3. **Adapter forecasting arm** (`diskgraph_runtime_adapter.py`): consume `task.autoregression_labels` + `task.yoy_labels`, place them on `hetero[train_table_name]` as `autoregression_labels` / `yoy_labels` tensor attributes (the names the forecasting encoder pops).
4. **Update `forecasting_gnn_module.py`** (431 LOC) to consume the new HeteroData shape. Model code unchanged where possible — pop pattern continues; new columns flow through the encoder.
5. **Fork `forecasting_lightning_data.py`** (152 LOC) to route through DG-native under `use_tt_ng_for_task`. Forecasting joins the discriminator's allowed-set; legacy arm stays until Phase 8a.
6. **Forecasting inference (prediction seed tables).** Reuse the `pred_tables/<job_id>/` machinery from Phase 4. `materialize_prediction_task.py` forecasting arm rewritten to call the shared pred-ingest entry point. Forecasting predict path uses the same adapter forecasting arm; AR/YoY labels are absent at predict time. ~60 LOC.
7. Tests green; metric parity with legacy on the forecasting e2e fixture for both training and predict.

**LOC estimate:**
- Sampler forecasting variant handling: ~220 LOC + ~180 LOC tests.
- Adapter forecasting arm: ~150 LOC + ~100 LOC tests.
- `forecasting_gnn_module.py` rewiring: ~100 LOC modified.
- `forecasting_lightning_data.py` fork: ~80 LOC + ~60 LOC tests.
- Forecasting arm of `materialize_prediction_task.py`: ~60 LOC + ~40 LOC tests.
- Phase 5 discriminator extension: ~30 LOC + ~30 LOC tests.
- **Total: ~640 LOC implementation + ~410 LOC tests ≈ 1050 LOC.**

**Acceptance:** PQ forecasting works end-to-end on DG-native for all four schema variants, both training and predict. Metric parity with legacy on the forecasting fixture.

#### Phase 9b — Custom-TT forecasting

**Problem:** custom-TT forecasting plans carry additional column shapes (segmented sweep buckets, custom weight columns from PR #255). Sampler reads these from the seed rowpack and routes them into the right `task.*` / `raw_features[*]` slots. Builds on Phase 9a PQ forecasting being stable.

**Three-layer scope split:**

1. **DG schema (already shipped — Phase 3.K):** `OutputMode::Forecasting { target_dtypes: Vec<DataType>, autoregression_dtype, yoy_dtype }` accepts vec-shaped TARGETs; `feat_columns` accepts arbitrary feat columns. No DG schema work needed here.
2. **Sampler emission (this phase):** handle custom-TT-specific column shape variants. Reuse `task.{autoregression_labels, yoy_labels}` slots from Phase 9a.
3. **Kumo-side multi-* consumption (Phase 10):** for Phase 9b, scope to single-target custom-TT forecasting. Multi-target / multi-entity consumption stays in Phase 10. Seed-row features, when present, remain part of the v1 contract rather than a deferred semantic unlock.

**Order (TDD):**
1. Write Phase 9b tests for custom-TT-specific cases: custom weight column (sidecar reuse from Phase 3.J), segmented-sweep semantics from PR #255, custom-TT vs PQ shape divergence on identical inputs.
2. **Sampler:** handle custom-TT forecasting's column shape variants. Segmented sweep buckets surface as additional `task.*` or `raw_features[*]` slots — decide in this phase's design step.
3. **Adapter:** extend forecasting arm to consume segmented-sweep slots when present.
4. **Custom-TT forecasting datamodule rewire:** parallel code path to PQ forecasting in kumo, dispatched by plan kind (custom-TT vs PQ). Same training + inference machinery from Phase 9a covers custom-TT predict.
5. Tests green; metric parity with legacy custom-TT forecasting (training + predict).

**Dependencies on PR #255 (+614 LOC on tt-ng branch):** verify what shipped in PR #255 still matches the schema we built in Phase 3.K. If PR #255 added forecasting-specific schema fields outside our `OutputMode::Forecasting` vec-ification, those need incorporation here or a follow-up issue.

**LOC estimate:**
- Sampler custom-TT extensions (segmented sweep, weight reuse): ~300 LOC + ~200 LOC tests.
- Adapter forecasting extension for segmented sweep: ~100 LOC + ~80 LOC tests.
- Custom-TT datamodule rewiring (training + predict arms): ~150 LOC + ~80 LOC tests.
- **Total: ~550 LOC implementation + ~360 LOC tests ≈ 910 LOC.**

**Acceptance:** custom-TT forecasting works end-to-end on DG-native at single-target parity scope, both training and predict. Segmented sweep semantics preserved. Metric parity with legacy custom-TT forecasting.

### Phase 8 — Legacy deletion sweep

Two sub-steps with separate go/no-go gates. Phase 8 cannot start until Phases 5 (node fork), 7 (LP), and 9 (forecasting) are stable in production — the targets are shared across all task types and cannot be deleted partially.

#### Phase 8a — Delete-behind-env-var (gated, reversible)

**Targets (all safely deletable because all task types have a DG-native path):**

Training-side:
- `kumo/train/datamodule/two_stage_node_lightning_data.py:196-250` — `_build_fkey_seeds`.
- `kumo/train/datamodule/two_stage_node_lightning_data.py:323-342, 434-447` — `_build_fact_feats` / 3-tuple plumbing.
- `kumo/train/two_stage_node_gnn_module.py:257` — `_attach_train_table`.
- `kumo/engine/diskgraph_sampler.py:1159-1177, 1224-1226, 1337-1345` — CSC convention adapter (no edge type goes through CSC anymore).
- `kumo/data/generic/_diskgraph_legacy_graphstore_bridge.py` — full deletion.
- `kumo/data/generic/diskgraph_train_table.py` — DataFrame-from-tt.parquet path.
- `_fkey_fetcher` second-NeighborSampler instantiation (`two_stage_node_lightning_data.py:130-133`).
- Forecasting-specific legacy paths in `forecasting_lightning_data.py`.

Prediction-side (replaced by the per-phase `pred_tables/<job_id>/` ingest in Phases 4, 7, 9):
- `kumo/data/generic/prediction_artifacts.py:174-243` — `build_prediction_table_edges_artifact`.
- `kumo/engine/diskgraph/prediction_artifacts.py` — `generate_diskgraph_prediction_artifacts` bundle JSON writer.
- `kumo-diskgraph/diskgraph-pybind/src/tt_executor.rs:3692-3820` — `_persist_prediction_artifact_bundle_native`.
- `_materialize_train_table_edges_from_manifest_bundle` (pred variant in `materialize_prediction_task.py`).
- `materialize_prediction_task.py:88-151, 632-654` — `_diskgraph_partition_table` (DG's internal sharding replaces it).
- `kumo/data/generic/diskgraph_batch_pred.py` — LP partition logic; replaced by sampler iterating the pred seed table directly.
- Sidecar artifacts: `fkey_cols.parquet`, `mask.parquet`, reverse-edge dirs (the in-memory `fkey` tensor concept the TT-NG migration explicitly kills).

DG-side Pass A col_stats redundancy (full deletion plan in `docs/tt-ng-architecture-and-migration.md` §11 step 8):
- `diskgraph-train-table/src/ttgen_stats.rs` — `ChunkStats` struct + `build_inline_col_stats` builder. Redundant now that every TT subdir runs through DG ingest at `train_tables/<tt_id>/<stage>/` and gets `col_stats.json` from the standard observer.
- `WelfordAccumulator` / `TDigestEstimator` / `HyperLogLog` field updates in `ttgen_executor.rs` and `custom_tt_chunk_processor.rs` — per-row hot path accumulation.
- `train_table_col_stats.json` emission. Consumers redirect to `<tt_id>/<stage>/col_stats.json` from the DG sub-index (via `discover_tts` + merged reader).
- Kumo's `seed_train_table_col_stats_from_manifest`.

  Hot-path win: ~50 cycles × N_columns per row eliminated. On a 100M-row TT, tens of seconds of CPU per worker.

**Order:**
1. Grep for remaining callers of each target. Confirm full coverage.
2. Delete behind `not use_tt_ng()` (gate, don't remove the env-off path). Confirm tests pass with env on AND env off.
3. Run with env-on in pre-prod for ≥ 1 release cycle.

**LOC:** ~-700 LOC Python deletion, all gated.

**Acceptance:** all tests green with both env states. DG-native is the only path for env-on; legacy path bit-identical for env-off.

#### Phase 8b — Flip default + remove env-off branches

**Order:**
1. Flip `KUMO_DISKGRAPH_TT_NG_ENABLED` default from off to on (one-line config change). Monitor for 1 release cycle.
2. Once stable: remove the env-off branches at every call site (the `else` arms in `materialize_training_task.py`, `auto_trainer.py`, etc.).
3. Delete `_tt_ng_dispatch.py` itself (per its own deletion note).

**LOC:** ~-300 LOC additional deletion.

**Acceptance:** legacy bridge fully removed. DG-native is the only path.

### Phase 10 — Kumo-side multi-entity / multi-target consumption (far future)

**Not on the near-term roadmap.** Phases 1–9 + 8 take us through every current kumo task type at parity, then delete the legacy bridge. The remaining large schema-expansion work is kumo-side **multi-target / multi-entity** consumption. Phase 10 captures that work for whenever it's prioritized — likely after Phase 8 has been stable for several release cycles and the ML team is ready to consume the broader schema end-to-end.

**Background:** Phase 3.K shipped DG-side schema capacity (multi-target, multi-entity, FEAT*). Seed-row features are part of the v1 semantics when present. What remains deferred is kumo-side consumption of **multi-target / multi-entity** shapes.

**Scope (kumo-side + small DG-side):**

1. **Multi-target consumption.** Adapter learns to stack `TARGET1..TARGETn` into a multi-dimensional `y` tensor on `hetero[train_table_name]`. Encoders downstream see y of shape `(B, N_targets)` instead of `(B,)`. Affects all three task types.
2. **Multi-entity consumption.** Adapter learns to handle `ENTITY1..ENTITYn` as separate task-bag node_id slots (today single ENTITY1 → single `seed_node_ids`).
3. **End-to-end test:** ML team's broader CustomTrainTableSpec fixture → DG ingest → seed extension carries multi-ENTITY / multi-TARGET → sampler routes by prefix → adapter consumes → trainer step succeeds and metrics are sensible.

**Dependencies:** Phases 4, 7, 9 all stable and 8a recommended (so this phase doesn't have to keep two adapter paths in sync).

**LOC estimate:** ~700 LOC implementation + ~580 LOC tests ≈ 1280 LOC.

**Acceptance:** ML team's broader multi-entity / multi-target contract runs end-to-end on DG-native and produces sensible training metrics on representative fixtures.

---

## 5. Follow-up PRs (outside the core sequence)

### Follow-up A — TT-NG dataplane persistence

**Problem:** `execute_tt_plan` writes the local `parent_index_path/train_tables/<tt_id>/` subdir. In pool deployments where activities run on different pods, the subdir must be uploaded to the dataplane after ingest completes, with `_SUCCESS` written last as the commit marker (matching PR-28723's discipline). PR-28723 explicitly skips TT-NG via `not use_tt_ng()`; TT-NG owns its own dataplane persistence.

**What it does:** after `execute_tt_plan` writes locally with `_SUCCESS`, sync the subdir to the dataplane. `JobConfig.diskgraph_task_dataset_bundle_dir` set to the dataplane location. Consumer activities (`materialize_train_task`, `train`) download the bundle to local before calling `open_with_tt`.

**Tests:**
- `test_tt_ng_subdir_synced_to_dataplane` — every file in the local subdir is present on the dataplane after `execute_tt_plan`.
- `test_tt_ng_subdir_success_written_last_on_dataplane` — `_SUCCESS` upload completes after all other files.
- `test_tt_ng_subdir_dataplane_idempotent` — re-running for the same `tt_id` overwrites cleanly.
- `test_tt_ng_cross_pod_handoff_smoke` — pod A produces locally → pod B reads from dataplane → `open_with_tt` succeeds.

**Implementation:** reuse PR-28723's `_sync_local_dir_to_remote(src_dir, dst_dir, excluded_basenames=["_SUCCESS"])` helper. Write `_SUCCESS` last via `fsutil.write_text(dataplane_success_path, marker)`. Boundary assertion: `assert fsutil.is_dataplane(JobConfig.diskgraph_task_dataset_bundle_dir)` post-condition.

**Scope:** kumo-side only, ~150 LOC + tests. Depends on Phases 1, 2, 4.

### Perf follow-up — TT writer Arrow builders

**Problem:** `ChannelBatchWriter` accumulates rows in nested `Vec<Vec<Value>>` and `Vec<Vec<Vec<Value>>>` buffers, one per ENTITY/TARGET/FEAT slot. Worst-case in-memory footprint between flushes is `O(64K × N_slots × max_list_len × sizeof(Value))` — multi-GB per writer thread on LinkArray with thousands of targets per entity × multi-target.

**Fix:** switch per-row Vec accumulators to Arrow's native typed builders directly per slot.
- Replace `target_value_list_bufs: Vec<Vec<Vec<Value>>>` with `Vec<ListBuilder<…>>` per slot.
- Replace `target_value_bufs: Vec<Vec<Value>>` with typed `Vec<Box<dyn ArrayBuilder>>` per slot.
- Same for `feat_value_bufs` / `feat_list_value_bufs`.

**Wins:** flat values + offsets instead of nested Vec headers (~10% memory), cache-friendly contiguous allocation, skip the `flush_value_buf` / `flush_value_list_buf` conversion at flush time.

**Scope:** ~150 LOC cleanup + ~50 LOC bench. No external API change (writer is internal to `diskgraph-train-table`). Non-blocking; the current pattern doesn't worsen the asymptote, it just multiplies by N_slots.

---

## 6. Risks and unknowns

Status as of Phase 3 completion:

1. **Does `index_metadata.json` (or any consumer of `edge_relations`) assume forward+reverse pairing?** Partly resolved. Phase 1's edge-phase guards and Phase 3.H's coordinated-layout fix held without breaking any consumer in the workspace tests. A full `edge_relations` consumer audit was not done — externals only live in the in-memory layout-computation Vec; `index_metadata.json` is unchanged on disk. Re-flag if any future consumer reads `edge_relations` and assumes pairing.
2. **Exact temp-file row format from `batcher.push_fkey`.** ✅ Resolved by Phase 1 — external-pkmap probe routes through `dst_maps` shard and the edge phase's existing probe path finds it transparently. No new row encoding needed.
3. **Whether `Uniform` sampling mode currently used in production for LP needs time filtering.** Open — relevant at Phase 6 (LP sampler primitives). Today's `engine.rs` Uniform path doesn't apply time filters. Phase 6 must decide.
4. **Schema confirmation: do TT-NG outputs carry `weight` and arbitrary feature columns?** ✅ Resolved. Weight as sidecar `weight.bin` (Phase 3.J). Feature columns as vec-shaped `feat_columns: Vec<ExtraColumnOutput>` writer slot + `push_feat_values` API (Phase 3.K). Seed-row features are part of the v1 semantics when present.

---

## 7. Out of scope

- Propagation of seed-table features through message passing ("make seed_table a graph participant" model change). Edges are materialized for this future option, but the model continues to pop the seed bag before convs.
- A general `ingest_one_table_into_index` API that registers a new table in the parent's `diskgraph_config.json`. Not needed — extensions live in sibling subdirs.
- HStack LP mode (`LinkExploded` / legacy). Explicitly unsupported under DG-native.
- Storage-format refactor to decouple pkmap from edges.bin or decolocate node attributes from edges.bin. Separate roadmap item.

---

## 8. Acceptance criteria

Each phase ships independently. Criteria are layered.

**End of Phase 3 ✅ SHIPPED — core architecture:**
- All Phase 1–3 tests pass (1509 cargo + 92 pytest at commit `b19a935a`).
- E2E correctness test passes: TT-NG → ingest → reader → sampler → runtime batch dict.
- Seed namespace never appears in `graphs.main.node_ids` or `edge_index_dict`; exclusively in `task.*` and (when features present) `raw_features[<seed_namespace>]`.
- Hop-1 temporal filter applies when seed has TIMESTAMP; no edge with `dst_time > seed_time` at hop 1.
- Parent index bytes unchanged after seed-table ingest.
- DG schema supports multi-target Link/Forecasting and `feat_columns` (Phase 3.K).

**End of Phase 5 — node prediction shipping at parity (training + inference):**
- All Phase 1–5 tests pass.
- `KUMO_DISKGRAPH_TT_NG_ENABLED=1` + a node-prediction TT runs one **training step** end-to-end on the e2e fixture (any of `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`, `MULTILABEL_CLASSIFICATION`, `MULTILABEL_RANKING`, `REGRESSION`).
- `KUMO_DISKGRAPH_TT_NG_ENABLED=1` + a node-prediction prediction seed table runs one **prediction step** end-to-end.
- Task-type discriminator routes LP and forecasting TTs to legacy (regression-tested).
- No Python artifacts of size O(num_seeds) for node-prediction TTs.
- `batch = next(dataloader)` works end-to-end for single-entity / single-target node prediction.

**End of Phase 7 — LP shipping at parity (training + inference):**
- All Phase 6–7 tests pass.
- `KUMO_DISKGRAPH_TT_NG_ENABLED=1` + LP TT runs one training step end-to-end on the e2e fixture for both `LinkGNNLightningModule` and `ShallowRHSLinkGNNLightningModule`.
- LP prediction runs end-to-end (MIPS index, top-K) on a pred fixture for both modules.
- `LinkPredTrainTable` phased out as past-visit exclusion source (Phase 6 collision check + stage-filter schema). Small eval-only cache may remain.
- LP metric parity with legacy GE LP.

**End of Phase 9 — forecasting shipping at parity (training + inference):**
- All Phase 9a + 9b tests pass.
- `KUMO_DISKGRAPH_TT_NG_ENABLED=1` + forecasting TT runs one training step end-to-end for PQ + custom-TT both.
- Forecasting prediction runs end-to-end on a pred fixture (PQ + custom-TT).
- Forecasting metric parity with legacy.

**End of Phase 8 — full legacy deletion:**
- All Phase 8a + 8b tests pass.
- Legacy bridge fully removed; `_tt_ng_dispatch.py` deleted; only the DG-native path exists.
- Cumulative net Python LOC change: ~-1000.
- Cumulative net Rust LOC change: ~+5500.

**End of Phase 10 — broader multi-* contract:**
- All Phase 10 tests pass.
- ML team's broader contract runs end-to-end on DG-native (multi-target, multi-entity).
- Adapter consumes those multi-* shapes and produces sensible training metrics.

### Cross-cutting invariants (apply at every phase boundary)

- No Python in-memory data structure proportional to `num_seeds` for the task type just shipped.
- Parent index bytes unchanged by any seed-table ingest.
- Stale extensions against an advanced parent fail loudly at open time.
- All new Rust code has unit tests; integration tests cover the end-to-end fixture for the task type just shipped.
- All existing tests in both repos continue to pass throughout.

---

## 9. TDD test specifications (appendix)

Every phase below lists tests that must be written and passing **before** the implementation work begins. Tests initially fail with a clear "not implemented" signal; implementation makes them pass.

### 9.0 Shared test fixtures

`kumo-diskgraph/test-fixtures/seed_tables/`:
- `tiny_parent_index/` — a finalized DG index with one entity table (`users`, ~100 rows) and one edge-bearing table (`transactions`, ~500 rows FK'd to users). Pre-ingested. Used as parent for all extension tests.
- `tiny_tt_node_pred.parquet` — synthetic TT-NG output for node prediction. ~50 rows. Columns: `ENTITY`, `TIMESTAMP`, `TARGET`, `weight`, `feature_a`, `feature_b`.
- `tiny_tt_lp_linkarray.parquet` — synthetic TT-NG output for LP LinkArray. ~50 rows. `TARGET` as `List<Int64>` of rhs PK values (avg 3 per row).
- `tiny_split.bin` — synthetic split assignment matching row counts.

Goal: <5s unit tests, large enough to exercise multi-shard ingest and multi-hop sampling.

### 9.1 Phase 1 tests: ingest plumbing

`kumo-diskgraph/diskgraph-ingest/tests/seed_table_ingest.rs`.

Unit tests:
1. `test_fkeyspec_serde_external_parent_roundtrip` — FKeySpec round-trip preserves fields; defaults preserve existing behavior.
2. `test_graphconfig_validate_accepts_external_fk` — external FK with out-of-config dst_table validates Ok; non-external errors.
3. `test_graphconfig_validate_rejects_external_fk_missing_src_column` — missing src_column still errors.
4. `test_ingest_skips_reverse_batcher_when_bidirectional_false` — no reverse RelationDef, no reverse temp files.
5. `test_ingest_mmaps_external_pkmap_for_resolution` — forward edges point to correct parent row indices.
6. `test_ingest_external_fk_key_failure_recording` — missing PK values recorded via KeyFailureState; no panic.
7. `test_ingest_parent_untouched` — every parent file checksum unchanged after ingest.
8. `test_ingest_atomic_publish_subdir` — `_SUCCESS` not written on mid-ingest panic; re-run succeeds.
9. `test_ingest_col_stats_observer_fires_for_seed_table` — col_stats includes feature columns.
10. `test_ingest_sort_keys_on_observation_ts` — `sort_keys-*.bin` exists and sorted by TIMESTAMP.

Integration tests:
11. `test_e2e_ingest_node_pred_tt` — full path produces correct subdir; reopens via `GraphIndexReader::open`.
12. `test_e2e_ingest_lp_linkarray_tt` — only `ENTITY1` produces FK edges; `TARGET1` (List) is a regular rowpack column; no FK for `TARGET1`.
13. `test_edge_relations_consumers_tolerate_asymmetric_relations` — `GraphIndexReader::open`, sampler's `_rel_id_to_edge_type` mapping, PR-28765 adapter's `edge_index_dict` all tolerate no-reverse-RelationDef.
14. `test_pkhashstore_loader_compat_matrix` — shard counts ∈ {1, 4, 16}; PK types ∈ {Int64, VarBytes, Uuid, Uuid256, Ip}; format version compatibility.

### 9.2 Phase 2 tests: merged reader

`kumo-diskgraph/diskgraph-index/tests/merged_reader.rs`.

1. `test_discover_tts_lists_finalized` — finalized tts returned in sorted order; incomplete excluded.
2. `test_open_with_tt_loads_parent_plus_one_extension` — parent tables at bare names, extension under `<tt_id>/<bare>`.
3. `test_open_with_tt_parent_untouched` — parent files byte-identical before and after open.
4. `test_merged_col_stats` — merged catalog exposes all column stats by namespaced key.
5. `test_merged_reader_fk_resolution_across_boundary` — seed.edges.bin yields valid parent row indices.
6. `test_open_with_tt_rejects_unknown_or_incomplete` — unknown tt_id errors loudly; subdir without `index_metadata.json` rejected as "not finalized".
7. `test_merged_edge_relations_namespaces_extension_src_type` — extension's seed→entity edge has namespaced src_type and bare dst_type; `reverse_of: None`.
8. `test_extension_namespacing_avoids_collision` — two TTs with same `train_seeds` name don't collide.

### 9.3 Phase 3 tests: sampler

`kumo-diskgraph/diskgraph-engine/tests/seed_sampler.rs`.

1. `test_sampler_walks_seed_edges_at_hop_0` — hop 0 reads from `seed.edges.bin`; hop 1+ from `parent.edges.bin`.
2. `test_sampler_temporal_filter_current_hop_1_behavior` — current implementation: every hop-1 edge has `dst_time ≤ seed_time`. This test documents the shipped behavior without settling the all-hop semantic question.
3. `test_sampler_does_not_emit_seed_as_node_type` — `graphs.main.node_ids` has no key for the seed namespace; `edge_index_dict` has no entry with seed-src.
4. `test_sampler_emits_runtime_batch_task_fields` — `task.{input_ids, tt_stage_node_ids, seed_node_ids, target, seed_timestamps?, weight?}` have correct lengths and values.
5. `test_sampler_emits_graph_dict_for_entities` — entity `node_ids` include seeds + k-hop neighbors deduplicated; `batch_dict` lengths match.
6. `test_sampler_emits_raw_features_for_seed_when_features_present` — `raw_features["<tt_id>/train_seeds"]` exists with seed feature columns when present, absent otherwise.
7. `test_sampler_emits_raw_features_for_entities` — entity `raw_features` exists with record batches for sampled rows.
8. `test_sampler_no_python_artifact_at_scale` — 1M-row seed batch, peak Rust memory bounded by `batch_size × constant`.

LP tests (deferred to Phase 6): `test_sampler_lp_target_list_expansion`, `test_sampler_lp_shallow_rhs_lhs_only`, `test_sampler_streaming_negative_sampling`.

### 9.4 Phase 4 tests: Python loader

`kumo/test/data/generic/test_dg_native_loader.py`.

1. `test_dataloader_returns_heterodata` — `next(dataloader)` returns HeteroData.
2. `test_dataloader_no_in_memory_artifacts` — no `fkey_cols.parquet` read, no DataFrame from `tt.parquet`.
3. `test_train_seeds_bag_has_expected_attributes` — `batch[train_table_name]` has `.tf`, `.batch`, `.time`, `.y`, `.weight`, `.n_id`, `.input_id`.
4. `test_entity_subgraph_present_in_batch` — entity types in `batch.node_types`; edge types in `batch.edge_types`.
5. `test_lp_shallowrhs_loader` — **deferred to Phase 7.**
6. `test_lp_standard_loader_with_linkarray` — **deferred to Phase 7.**
7. `test_model_forward_pass_completes` — forward pass with no exceptions; output shape matches target.
8. `test_model_pops_train_table_unchanged` — `x_dict[train_table_name]` pop pattern works.
9. `test_env_var_gates_dg_native_path` — all four combinations of env var × wheel produce the right path.
10. `test_runtime_batch_contract_with_pr28765_adapter` — pin runtime-batch shape against PR-28765's `runtime_batch_to_hetero` contract.

### 9.5 Phase 3 acceptance: end-to-end correctness

`kumo-tests/diskgraph/test_seed_node_table_e2e.py`.

1. `test_e2e_node_prediction_runtime_batch_shape` — full pipeline through `make_runtime_batch_iterator`; runtime batch has every adapter-required field with correct lengths.
2. `test_e2e_temporal_no_leak_hop1` — no edge with `dst_time > seed_time` at hop 1.
3. `test_e2e_no_seed_in_topology` — `graphs.main.node_ids` has no seed-namespace key; `edge_index_dict` has no seed-src edges.
4. `test_e2e_multiple_tts_per_parent` — two TTs ingested into the same parent open independently; `discover_tts` returns both; namespacing avoids collisions.

LP and forecasting e2e tests live in their own phases.
