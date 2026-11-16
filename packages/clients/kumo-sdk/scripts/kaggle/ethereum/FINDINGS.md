# `kaggle_ethereum_address_scam_30gb` on the SDK's local backend

Row 1 of [`kaggle-fixture-matrix.md`](../../../kaggle-fixture-matrix.md), run end
to end in-process — Snowflake catalog reads, DiskGraph ingest, `kumo_core`
AutoML, batch scoring — with no Kumo service, no Temporal, no Kubernetes.

**Result: it works, at the full published scale.** All 380,566,661 transactions
and 3,074,622 addresses, on 4 vCPU / 15 GB RAM / one L4.

Two things were needed to get there. The fixture's PQL as published cannot
select any rows to score, which is a property of the fixture rather than a
backend defect (§1); and the durable snapshot copy has to be skipped or the
index does not fit, which the backend had no way to express and now does (§3).

## Runs

|                    | smoke           | part                | **full**            |
| ------------------ | --------------- | ------------------- | ------------------- |
| transactions       | 5,000,000       | 40,000,000          | **380,566,661**     |
| addresses          | 82,339          | 3,074,622           | 3,074,622           |
| labeled / held out | 65,834 / 16,505 | 2,459,254 / 615,368 | 2,459,254 / 615,368 |
| train              | 118.9 s         | 1,792.1 s           | 4,929.4 s           |
| predict            | 1.6 s           | 17.2 s              | 38.2 s              |
| test AUROC         | 0.9901          | 0.9764              | 0.9934              |
| test AP            | 0.9293          | 0.6922              | 0.7770              |
| held-out AUROC     | 0.9844          | 0.9794              | **0.9937**          |

Held-out AUROC is computed against labels blanked out of the graph entirely
(§1), so it is measured on rows the model never saw, using the rank identity
rather than `sklearn` (§2).

Only the transactions grow between `part` and `full` — the address table is
already complete at `part`. That is why training grows ~2.75× rather than 9.5×:
the entity count, and so the AutoML cost, is unchanged. Ingest is what scales.

## 1. The published PQL cannot score anything

`PREDICT addresses.is_scam FOR EACH addresses.address` is a **static** node
target. Compiling it in prediction mode yields:

```json
"filter": {"compare": {"alias": "col_1", "op": "is_null",
                       "rhs": {"constant": {"int": 0}}}}
```

The prediction set is *the entities whose target is NULL* — train on labeled
rows, score unlabeled ones. The fixture labels all 3,074,926 addresses
(`n_null_label = 0`), so `generate_prediction_table` raises
`RuntimeError: prediction query produced no rows`.

This is not an anchor problem. `probe_ethereum_predict.py` re-runs TT generation
at four anchors spanning 2014–2026 against an already-ingested snapshot; all
four return `rows=0`.

**Workaround:** `--holdout-pct 20` blanks `is_scam` for a deterministic slice
(`MOD(ABS(HASH(address)), 100) < 20`), keeping true labels in a
`*_TRUTH` side table. This restores the intended shape and doubles as a clean
generalization check.

**This will hit matrix rows 2, 3, 4 and 10 the same way** — every static-column
`PREDICT` (`iot23.is_malicious`, `bccc.label`, `iot23.detailed_label`,
`yambda.played_ratio_pct`). The temporal/aggregate rows are unaffected.

## 2. `scikit-learn` is missing from the fat wheel

Every run logs `Optuna search analysis failed` /
`ModuleNotFoundError: No module named 'sklearn'`. Non-fatal — training and refit
complete — but see below for what is actually lost.

Neither `kumoai_studio` nor `kumo_core` imports `sklearn`. The chain is:

1. `kumo_core/training/orchestration.py:406` calls `searcher.analyze(...)`.
1. `kumo_core/automl/optuna_searcher.py:107` calls optuna's
   `plot_param_importances`.
1. That needs `sklearn.ensemble.RandomForestRegressor` via fanova.
1. optuna declares it as `Requires-Dist: scikit-learn>=0.24.2; extra == "optional"` — an **extra**, so a plain `optuna` install omits it.
1. `scripts/build_fat_wheel.sh:621` unions component requirements and skips
   extras outright:
   ```python
   # Extras are not installed, so their requirements do not apply.
   if 'extra ==' in requirement:
       continue
   ```

So the bundle never learns about it. Fixed by declaring `scikit-learn>=1.7.2` on
`kumo-core`, matching the root package's `install_requires`:
[kumo-ai/kumo#30507](https://github.com/kumo-ai/kumo/pull/30507).

**What is lost is more than the name suggests.** `analyze()` wraps all plotting
in one `try/except ... return`, and `plot_optimization_history` runs *before*
the call that raises. So the early return discards **all three** artifacts —
`optimization_history.png`, `param_importances.png` and the slice plots — not
just parameter importances.

## 3. The durable snapshot copy had no off switch

A local run stored the DiskGraph index twice: under `scratch` where it is built,
and under `artifacts` where `checkpoint` publishes it so a later run restores
instead of re-ingesting. `LocalSnapshotStore.checkpoint` does that
unconditionally (`copy_tree` -> `replace_tree`), and `Storage` offered no
way to decline. At full scale the second copy is ~39 GB, which is what put the
fixture out of reach of the root volume.

**Now the copy is gone by default.** `Storage.artifacts` is an
`ArtifactLocation` -- the durable root and its publication policy in one
object, so the two cannot name different places. `FilesystemArtifactLocation`
defaults to `PublishMode.SYMLINK`: the durable path is a link to the index in
`scratch` rather than a second copy of it, so no configuration is needed to
avoid the 39 GB. Ask for the old behaviour only when the index has to outlive
`scratch`:

```python
kumo.LocalBackend(storage=kumo.Storage(
    scratch='/mnt/localssd/kumo-run/scratch',
    artifacts=kumo.FilesystemArtifactLocation(
        root='/mnt/localssd/kumo-run/artifacts',
        publish_mode=kumo.PublishMode.COPY)))
```

Rebuilds stay cheap. `restore` reports the scratch index whenever it carries a
`_SUCCESS`, so an unchanged graph skips ingest exactly as the durable path
would: what is skipped is the copying, not the caching. Nothing else in the flow
checks whether `scratch` is warm, so a literal no-op would have re-ingested
every run.

Verified end to end at smoke scale: `artifacts/` holds 904 KB — predictions and
task dataset only, no `_SUCCESS` anywhere — against 564 MB of scratch, and a
second run left the snapshot's `ingest_stats.json` mtime untouched.

Trade: nothing survives a wiped `scratch`. It is for local development, not for
a container whose scratch dies with it. `run_ethereum_local.py --no-publish-snapshots` selects it.

Measured at full scale, with the durable copy in place:

|                        |              |
| ---------------------- | ------------ |
| scratch index          | 39 GB        |
| **peak run footprint** | **59.0 GiB** |
| free at peak           | 61.3 GiB     |

**Two things mattered more than the copy.** There is no `source_tables`
directory — DiskGraph streams from presigned URLs, so 26 GB of staged Parquet
costs *zero* local disk. And the peak (59 GiB, reached ~9 min in during ingest)
sits well above the 39 GB steady state, so size headroom for the transient, not
the resting footprint.

## 4. Storage and warehouse

The root volume had 46 GB free — not enough. The runs use the **EC2
instance-store NVMe** (`/dev/nvme1n1`, 233 GB, already ext4 with label
`localssd`, simply not mounted):

```
sudo mount /dev/nvme1n1 /mnt/localssd      # holds pre-existing user data
```

It is instance storage: **contents are lost on stop/start**, which is correct
for `scratch` but means `artifacts` is not durable across an instance stop.

`CURRENT_WAREHOUSE` defaulted to `KUMO_XS` because `~/.config/snowflake/ config.toml` sets none. `KUMO_L` (8×) on the 380M-row CTAS:

|               | XS                      | L         |
| ------------- | ----------------------- | --------- |
| 40M-row CTAS  | 365 s                   | —         |
| 380M-row CTAS | ~3,470 s (extrapolated) | **402 s** |

**8.6×** — essentially nominal, so the step is compute-bound rather than
S3-bound. `KUMO_M`/`KUMO_XL`/`KUMO_2XL` are also available.

`SnowflakeConnector(warehouse=...)` is **inert** on the local backend; reads run
on the registered session. The warehouse must therefore be selected inside the
session *factory* — the backend re-runs it — not once at startup.

## 5. Fixture corrections

Against the matrix's "Dataset roots and graph contract" row:

- **`addresses.address` is not unique.** 3,074,926 rows, 3,074,622 distinct —
  304 duplicates. Deduplicated by most-recent `activity_end_ts` before use as a
  primary key.
- **`labeled_address_ids` is unusable and unnecessary.** Transactions carry raw
  address strings (`from_address`, `to_address`), so it contributes no edge; and
  its `node_id` holds unsigned 64-bit hashes (e.g. `13020063922375618244`) that
  exceed `int64` max. Dropped from the graph.
- The parts are **not** day-contiguous despite the `transactions_daily_part-*`
  naming — each spans 2015-08 to 2025-08. One part is a representative ~10%
  sample, not a prefix.
- `description` is dropped: free text naming the scam type, so a label leak as
  well as a text-encoder dependency. `tx_hash` is dropped as row-unique noise.

The graph is two tables with **two foreign keys into the same destination**
(`from_address` and `to_address` → `addresses`), which the portable-graph
conversion handles correctly — edges key on `(src, fkey, dst)`.

## 6. Open question: leakage

0.99 AUROC on a 1.78%-positive problem, rising with scale (AP 0.692 → 0.777).
Plausible for this dataset, but `activity_start_ts` / `activity_end_ts` summarize
an address's entire activity window, including time after the label was
assigned. Worth an audit before quoting these as benchmark numbers. This is a
property of the fixture, not of the backend, and does not affect the end-to-end
result.

## 7. Snapshot identity is wider than the data

A snapshot lives at `<graph key>/<data version>`:

- **graph key** — fingerprint of the portable graph: table names, source table
  names, every column's name/dtype/stype, foreign keys, time column, *and each
  table's `connector_id`*.
- **data version** — fingerprint of per-table versions, each
  `sha256("DB.SCHEMA.TABLE|LAST_ALTERED|ROW_COUNT")[:16]`.

Two consequences found by accident.

**The connector ID includes `warehouse`, which is inert for local reads.**
Changing `SnowflakeConnector(warehouse=...)` from `KUMO_XS` to `KUMO_L` — a
field the local backend never uses, since reads run on the registered session —
moved the smoke graph key from `4f301a28182add30` to `aedfd571232c7a12` while
the data version stayed `ef6e278051268ea4`. Same data, provably; whole index
rebuilt, durable copy orphaned. At full scale that is a ~45 minute re-ingest for
editing a string with no effect on the data.

**`restore` does not check whether scratch already holds the snapshot.** It
copies durable → scratch unconditionally, then verifies. On any re-run with warm
scratch that is a redundant full-size copy — 39 GB at full scale — and it is
a redundant full-size copy of an index that is already there.

Also worth knowing: `train_tables` are written *inside* the scratch snapshot
directory but *after* `checkpoint`, so they are never in the durable copy. A
restore gives back the index, not a fully warm run.

*Not tested, but implied by the version formula:* a **view** has a static
`LAST_ALTERED` and a null `ROW_COUNT`, so a view-backed source would hash to the
same version forever and never invalidate its snapshot. Relevant because a view
over the external stage is otherwise an attractive way to skip materializing
26 GB into Snowflake.

## Reproducing

```bash
# One-time: external stage over the S3 fixture, via the PUBLIC_DATASET
# storage integration (allowed location is the whole bucket).
source scripts/_env.sh  # resolves the monorepo SDK and root .venv
"$KUMO_SDK_PYTHON" scripts/kaggle/ethereum/probe_ethereum_fixture.py

# Materialize Snowflake tables. Scales: smoke | part | full.
"$KUMO_SDK_PYTHON" scripts/kaggle/ethereum/build_ethereum_tables.py \
    --scale full --holdout-pct 20 --warehouse KUMO_L

# Run end to end on the instance-store NVMe.
"$KUMO_SDK_PYTHON" scripts/kaggle/ethereum/run_ethereum_local.py \
    --scale full --no-publish-snapshots

# Or chain both unattended, sampling free disk once a minute.
scripts/kaggle/ethereum/run_full_when_ready.sh
```

Objects created in Snowflake (account `TLB22710`):

|                       |                                                                              |
| --------------------- | ---------------------------------------------------------------------------- |
| `HYU.KAGGLE_FIXTURES` | external stage `ETHEREUM_FRAUD_30GB`, file formats `JSON_FF` / `PARQUET_FF`  |
| `HYU.ETH_FRAUD`       | `TRANSACTIONS_{SMOKE,PART,FULL}`, `ADDRESSES_{...}`, `ADDRESSES_{...}_TRUTH` |

`TRANSACTIONS_FULL` costs ~26 GB of Snowflake storage. The local backend's
Snowflake reader reads *tables*, not stages, which is why the staged Parquet has
to be materialized first.

## Scripts

| file                        | purpose                                             |
| --------------------------- | --------------------------------------------------- |
| `probe_ethereum_fixture.py` | stage the S3 fixture; manifest and inferred schemas |
| `probe_ethereum_stats.py`   | class balance, key uniqueness, part-file day ranges |
| `probe_ethereum_predict.py` | why the prediction table is empty (§1)              |
| `build_ethereum_tables.py`  | materialize Snowflake tables at a chosen scale      |
| `run_ethereum_local.py`     | the end-to-end run                                  |
| `run_full_when_ready.sh`    | unattended full-scale driver with disk sampling     |
