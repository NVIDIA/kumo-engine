# `kaggle_bccc_flow_attack_type_20gb` on the SDK's local backend

Row 3 of [`kaggle-fixture-matrix.md`](../../../kaggle-fixture-matrix.md), run
end to end in-process, the same way [row 1](../ethereum/FINDINGS.md) and
[row 2](../iot23/FINDINGS.md) were. The first **multiclass** fixture.

**Result: the pipeline works.** Two tiers complete end to end, and at `small`
the model genuinely separates most of the class tail (§4). Two caveats dominate
everything else here: the reported `test_acc` is a constant predictor's score at
both scales because the temporal split is really a *class* split (§3), and the
run surfaced a **`kumo_core` defect that silently corrupts column statistics**
for 44% of this fixture's columns (§6), plus one unexplained symptom (§7).

## Runs

|                           |              smoke |                   small |
| ------------------------- | -----------------: | ----------------------: |
| flows / hosts             |     23,998 / 6,101 |    **481,767 / 42,568** |
| classes present           |           11 of 14 |                13 of 14 |
| held out for scoring      |              4,872 |                  96,640 |
| train                     |            195.5 s |             2,474.5 s † |
| predict                   |              4.2 s |                  35.5 s |
| DiskGraph index           |              55 MB |                  797 MB |
| `test_acc` (reported)     |             0.9362 |                  0.9409 |
| — test split Benign share |             0.9361 |                  0.9409 |
| **held-out accuracy**     |             0.8988 |              **0.9896** |
| **held-out macro-F1**     | 0.1052 (9 classes) | **0.6731 (11 classes)** |

† **Contended.** A row-2 `part`-tier run belonging to another session shared
the 4 vCPUs for the whole of both tiers, and part-way through `small` that
session wiped `/mnt/localssd` — which is why BCCC now uses its own root under
`/mnt/localssd/kumo-bccc`, and why some post-hoc artefacts (§7) were
unrecoverable. The `small` figures survived because they were computed before
the wipe and are internally exact: per-class supports sum to 96,640, the TRUTH
table row count, and prediction rows to 96,640 × 13 = 1,256,320. Treat every
timing here as an upper bound, not a clean measurement.

Held-out figures are measured against labels blanked out of the graph (§2), on
a uniform hash sample the model never saw. Row 1 and 2's `_auroc` helper does
not transfer — see §5.

## 1. The fixture is not shaped like the matrix says

| claim                       | measured                                                                                                                            |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 48,590,026 rows             | **48,182,782** `network_flows` + **407,244** `hosts` = 48,590,026 — the matrix figure is the **sum**, as in rows 1 and 2            |
| key `network_flows.flow_id` | **not unique**: 48,182,782 rows → 48,120,094 distinct (62,688 collisions, 0.13%). Deduplicated on the way in, as ethereum's key was |

- **325 columns**, against iot23's 21. 19.86 GB Parquet but **106.02 GB
  uncompressed** — 2.2 KB/row against iot23's ~43 B. Width, not row count, is
  what makes this fixture expensive: the raw typed copy took 1,479 s on KUMO_L.
- `hosts` is 407,244 rows of a single column (`host_id`) — pure structure, no
  features, same as iot23.
- `timestamp` is **microseconds**; pinned exactly because `flow_id` embeds the
  same instant as text.
- **22 columns infer as VARIANT** — float statistics written as JSON strings in
  some part files — and `Infinity` is real data, not a parse artefact:
  `cov_payload_bytes_delta_len` is non-finite in **70.7%** of rows. Both are
  handled in the cast. Note Snowflake's `NaN = NaN` is *true*, so `v != v` does
  not filter them; the build uses `BETWEEN -1e308 AND 1e308`.

### The class tail is extreme

14 classes: `Benign` 89.42%, `DDoS_HOIC` 4.50%, `DoS_HULK` 3.79%, then a long
tail down to **`Infiltration` at 5 rows in 48.1M**. The four rarest classes
total 879 rows.

**Only the full fixture is a 14-class problem.** `smoke` is 11-class and `small`
is 13-class — the rare classes simply are not in a sample. Any comparison across
tiers is comparing different tasks.

## 2. Static target, third confirmed instance

`PREDICT network_flows.label` compiles to `WHERE label IS NULL`, the fixture
labels every flow, so batch prediction as published selects nothing. Same fix as
rows 1 and 2: a deterministic holdout with a `*_TRUTH` side table.

Row 1 §1 predicted rows 2, 3, 4 and 10 would all hit this. **Three down.** Row 4
is the same fixture family as row 2 and row 10 is `yambda`; both should be
assumed affected.

## 3. The temporal split is a *class* split — NEW, and worse than row 2

This is the finding that governs how every number here should be read.

All **30 capture sessions are single-label** — `N_LABELS = 1`, purity
1.000000, measured not approximated — and captures are time-contiguous. So a
temporal split does not sample the problem; it partitions by capture, and
therefore by class.

Read from `published-model/manifests/input.json` (`stage_counts`,
`stage_seed_times`) and recovered independently from Snowflake over each
stage's time window, at both scales:

| tier  | stage | window                    |    span |    rows | classes | composition                       |
| ----- | ----- | ------------------------- | ------: | ------: | ------: | --------------------------------- |
| smoke | train | 02-14 07:28 → 02-28 12:03 | 14.19 d |  15,292 |      10 | Benign 87.76%                     |
|       | val   | 02-28 12:03 → 03-02 08:28 |  1.85 d |   1,921 |   **1** | Benign 100%                       |
|       | test  | 03-02 08:28 → 03-02 19:18 |  0.45 d |   1,910 |   **2** | Benign 1,788 (93.61%), Bot 122    |
| small | train | 02-14 07:28 → 02-28 11:51 | 14.18 d | 307,948 |      12 | Benign 87.51%                     |
|       | val   | 02-28 11:51 → 03-02 08:14 |  1.85 d |  38,577 |   **1** | Benign 100%                       |
|       | test  | 03-02 08:14 → 03-02 19:28 |  0.47 d |  38,599 |   **2** | Benign 36,317 (94.09%), Bot 2,282 |

(The manifest's `stage_counts` are 15,293/1,922/1,911 and 307,949/38,578/38,600;
the recovery is one row short per stage because the window bounds are inclusive
of a seed instant that the `>=`/`<=` recovery straddles. Immaterial.)

Two consequences:

- **`test_acc` is a constant predictor's score.** The test split's Benign share
  is 1,788/1,910 = 0.9361 at smoke and 36,317/38,599 = 0.9409 at small, against
  reported `test_acc` of 0.9362 and 0.9409. The metric cannot see 9 of the 11
  classes at smoke, or 11 of the 13 at small. `test_refit_acc` equals `test_acc`
  to four decimals at both scales, which is what two collapsed models look like.
- **Model selection is blind.** Validation is 100% Benign, so *any* metric
  computed on it is degenerate — `val_acc` reads 1.000 throughout training for a
  model that predicts one class.

The pathology is **scale-invariant** — it is driven by capture boundaries, not
sample size — so it will not improve with a bigger tier.

This is row 2 §5 amplified. There, a temporal split skewed the class balance to
99.96% one class; here it removes classes from the split entirely. Any fixture
whose label composition is time-correlated will do this, and every network
capture dataset is time-correlated by construction.

**What to trust:** the held-out score, which is a uniform hash sample over the
whole fixture — 96,640 rows and 11 classes at `small`.

## 4. The smoke-scale collapse was real, and it resolved with scale

At smoke the model predicted **Benign for every one of the 4,872 held-out
rows**: accuracy 0.8988, exactly the majority rate, macro-F1 0.1052.

At `small` it does not: accuracy **0.9896** against a **0.8940** majority
baseline (the held-out Benign share), and **macro-F1 0.6731 over 11 classes with
support**. So the collapse was a small-sample artefact, not a permanent property
of the fixture.

The per-class table is the part worth reading, because the macro mean hides two
quite different failures:

| class               | support | predicted | precision | recall |     F1 |
| ------------------- | ------: | --------: | --------: | -----: | -----: |
| Benign              |  86,394 |    86,990 |    0.9931 | 1.0000 | 0.9966 |
| DDoS_HOIC           |   4,313 |     4,313 |    1.0000 | 1.0000 | 1.0000 |
| DoS_HULK            |   3,687 |     3,688 |    0.9997 | 1.0000 | 0.9999 |
| DDoS_LOIC_HTTP      |     559 |       560 |    0.9982 | 1.0000 | 0.9991 |
| Brute_Force_SSH     |     399 |       399 |    1.0000 | 1.0000 | 1.0000 |
| DoS_Golden_Eye      |      57 |        57 |    1.0000 | 1.0000 | 1.0000 |
| DoS_Slowloris       |      17 |        14 |    1.0000 | 0.8235 | 0.9032 |
| DoS_SlowHTTP        |     209 |       619 |    0.3376 | 1.0000 | 0.5048 |
| **Bot**             |     594 |     **0** |    0.0000 | 0.0000 | 0.0000 |
| **Brute_Force_FTP** |     410 |     **0** |    0.0000 | 0.0000 | 0.0000 |
| Brute_Force_Web     |       1 |         0 |    0.0000 | 0.0000 | 0.0000 |

Seven of eleven classes are at or near F1 1.0. The macro mean is dragged down by
three zeros, and the two that matter have *different* causes — recoverable by
arithmetic on the predicted-minus-support column, since it sums to zero:

- **`Bot` (−594) is never in training.** Its capture, `friday_02_03_2018_bot`,
  is the last in the fixture, so §3 puts every Bot row in the *test* window.
  The model cannot learn a class it never sees; all 594 are scored Benign, which
  with `Brute_Force_Web` (−1) and one `DoS_Slowloris` accounts for Benign's
  +596 exactly. This is §3's pathology showing up as a per-class zero.
- **`Brute_Force_FTP` (−410) is confused with `DoS_SlowHTTP` (+410).** The match
  is exact: `DoS_SlowHTTP` has 209 support but 619 predictions, and 619 − 209 =
  410\. That is a genuine model error rather than a split artefact — FTP brute
  force has 1,557 training rows.

Macro-F1 0.67 with accuracy 0.99 is the expected shape for a long tail, but on
this fixture two of the three zeros are structural, not statistical.

## 5. Multiclass changes the plumbing, not just the metric — NEW

- **There is no AUROC or AP for multiclass at all.** The supported set is `acc`,
  `precision`, `recall`, `f1` (all macro-averaged) and `mrr`.
- **The default plan tunes on `acc` alone.** On a fixture with an 89% majority
  class that selects the model which ignores the tail — precisely the smoke
  result in §4.
- **Predictions come back long**, not wide: `ENTITY | PREDICTED | CLASS | SCORE`, one row per entity per class, capped at `min(num_classes, 20)`. Both
  tiers are under the cap, so all classes are returned: 4,872 × 11 = 53,592 rows
  at smoke and 96,640 × 13 = 1,256,320 at small, both exact. `PREDICTED` marks
  the argmax row — it is *not* ground truth, despite reading like it. Rows 1–2
  returned `True_PROB`/`False_PROB` columns, so their `_auroc` helper does not
  transfer; this run computes per-class precision/recall/F1 and a macro-F1
  instead, and detects the wide and long shapes rather than assuming one.
- **`num_entities_predicted` reports the row count, not the entity count** —
  53,592 for 4,872 entities at smoke. Worth knowing before quoting it.

## 6. `_column_stats` silently corrupts statistics — NEW, verified, unreported upstream

`kumo_core/pquery.py::_column_stats` matches stat-type tokens by **substring
against the whole key**, and the key includes the column name. The longest
`ColStatType` token appearing anywhere wins, so a column whose *name* contains
one loses its real statistics.

The keys look like `PACKETS_COUNT__kumo.numerical__MEAN__False`, and the lookup
is:

```python
for stat_type in sorted(ColStatType, key=lambda value: len(value.value),
                        reverse=True):
    if stat_type.value in key:
        result[stat_type.value] = ...
        break
```

`PACKETS_COUNT` contains `COUNT` (5 characters), which is tested before `MEAN`
(4), `MIN`, `MAX` and `STD` (3). So all of those are filed under `COUNT`, each
overwriting the last.

Read back through `_column_stats` against the real smoke index, alongside the
raw values from the same file:

| column                             | parsed                                               | actual                                                        |
| ---------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| `DURATION` *(control, no token)*   | `MEAN 24.160, MIN 0.0, MAX 15216.583, STD 115.826`   | identical — correct                                           |
| `PACKETS_COUNT`                    | `COUNT 224.108`; `MEAN`/`MIN`/`MAX`/`STD` **absent** | `COUNT 23998, MEAN 16.961, MIN 1.0, MAX 23240.0, STD 224.108` |
| `PAYLOAD_BYTES_MEAN`               | **`MEAN 136.172`**, `MEDIAN 67.343`                  | **`MEAN 99.351`**, `STD 136.172`, `MIN 0.0`, `MAX 2820.452`   |
| `MAX_HEADER_BYTES` *(categorical)* | `MEAN 1.163, MIN −20.0, MAX 4.275`                   | **has no numerical statistics at all**                        |

Only `PACKETS_COUNT` raises — `PredictiveQueryPlanError: encoder inference for 'PACKETS_COUNT' requires MEAN` — because the collision leaves a *required* stat
missing. That is the benign case: it stops.

**`PAYLOAD_BYTES_MEAN` is the damaging one.** Its parsed mean, 136.172, is
actually its standard deviation; the true mean is 99.351. Nothing raises, the
encoder is selected against the wrong statistic, and the column is normalised
against it. `MAX_HEADER_BYTES` is worse in kind: a categorical column with no
numerical statistics whatsoever parses to four invented ones, manufactured from
its `CATEGORY_COUNTS` payload.

**Blast radius: 142 of 323 columns (44%) here** — `COUNT` 29, `MEAN` 23, `STD`
23, `MEDIAN` 23, `MAX` 22, `MIN` 22. Rows 1 and 2 never tripped it because no
ethereum or iot23 column name contains one of the 36 `ColStatType` tokens.
Feature-engineered datasets are full of them, and BCCC names every per-flow
statistic after the statistic it is, so it is the first fixture to reach a bug
that was always there.

Worked around here by renaming the columns in the tier tables
(`build_bccc_tables.stat_safe_name`: `count→cnt`, `median→mid`, `mean→avg`,
`min→lwr`, `max→upr`, `std→sdv`, longest-first so no rewrite creates a new
token). 157 of 323 columns are renamed; afterwards none contains a token and all
323 names stay distinct. **This is a deviation from the fixture's published
column names** and should be accounted for in any comparison with a published
BCCC baseline.

The fix belongs upstream — match the stat token against the key's suffix rather
than the whole key. That one-line change was **not** written or tested here, so
it is a proposal, not a verified patch.

## 7. Model-plan overrides are silently dropped — NEW, verified, mechanism not isolated

The run passes `{'training_job': {'metrics': [acc, precision, recall, f1, mrr], 'tune_metric': 'f1'}}` to `kumo.Trainer(...)`, and the log confirms it. But:

- `resolved-model-plan.json` reads `metrics: ['acc'], tune_metric: 'acc'` — the
  default;
- only `acc` and `loss` were ever computed; no `f1`, `precision`, `recall` or
  `mrr` appears anywhere in the run.

What is **directly measured** here is the symptom: across both tiers the only
metrics the SDK ever reports are `test_loss`, `test_acc`, `test_refit_loss`,
`test_refit_acc`, `train_loss` and `train_acc`. No `f1`, `precision`, `recall`
or `mrr` appears anywhere, at either scale.

What is **not** established is where they are lost. The keys are in the right
place — `metrics` and `tune_metric` live under `training_job`, and
`ModelPlan.update_from_yaml` raises `ExtraFieldsError` on an unknown top-level
key rather than ignoring it, so a misplaced override would have failed loudly
rather than silently. Beyond that the drop point was not isolated, and the
`resolved-model-plan.json` that would settle it was lost when the scratch volume
was wiped (see the note under Runs). **Stated as an open, reproducible symptom,
not a diagnosis.**

It matters on this fixture: tuning on accuracy against an 89% majority is what
selects the degenerate model in §4.

## 8. Scale ceiling: time, not disk — measured at two points

Disk is comfortable. Index size per row *falls* with scale as fixed overheads
amortize:

| tier  |   flows |  index | per row |
| ----- | ------: | -----: | ------: |
| smoke |  23,998 |  55 MB | 2.29 KB |
| small | 481,767 | 797 MB | 1.65 KB |

Marginal rate between the two is **1.62 KB/row**, so the full 48.12M rows
extrapolate to **~78–79 GB** of index — comfortable inside 210 GB free even
allowing the ~1.5× ingest transient row 1 measured. (An earlier ~110 GB estimate
from the smoke point alone was an over-estimate and is retracted.)

Time is the ceiling. 20.1× the rows cost 12.7× the training time (195.5 s →
2,474.5 s), so the marginal cost between the two measured points is

```
(2,474.5 − 195.5) / (481,767 − 23,998) = **4.98 ms per row**
```

| tier  |   flows | est. train | basis        |
| ----- | ------: | ---------: | ------------ |
| smoke |  23,998 |    195.5 s | **measured** |
| small | 481,767 |  2,474.5 s | **measured** |
| part  |   2.41M |     ~3.3 h | extrapolated |
| half  |  24.06M |      ~33 h | extrapolated |
| full  |  48.12M |  **~66 h** | extrapolated |

**This answers the question the fixture was expected to settle, and the answer
is no.** BCCC has 6.8× *fewer* rows than iot23, so full scale looked reachable
where iot23's was not. It is not, because per-row cost is 9× higher: 4.98 ms
here against iot23's 0.553 ms (1,801 s / 3,254,288 flows). 318 encoded columns
against 13 more than cancels the row-count advantage, and BCCC's ~66 h estimate
is *worse* than iot23's ~50 h. Time is the binding constraint on both, and disk
on neither.

Everything past `small` is extrapolated from two points assuming linearity, and
both measured points were contended. `part` was launched and deliberately
stopped — another agent had begun row 4 on this box and had already wiped the
scratch volume once mid-run, so a 3-hour contended job was not worth the shared
CPU. **Largest measured tier: `small`.**

Note §1: full scale is also the *only* tier that is actually the published
14-class problem.

## 9. Leakage: two columns dropped, both confirmed exactly

| column       | why                                                     | measured                                                       |
| ------------ | ------------------------------------------------------- | -------------------------------------------------------------- |
| `is_attack`  | the target's binary collapse — literally row 2's target | `is_attack != (label <> 'Benign')` in **0 of 48,182,782 rows** |
| `capture_id` | names the scripted attack session                       | all 30 captures single-label, purity 1.000000                  |

`capture_id` is embedded in `flow_id`, and the capture names literally spell the
attack — `wednesday_21_02_2018_hoic_part2`. As in row 2, this is **inert**:
`published-model/model/encoders.json` shows 318 encoded columns on
`network_flows` and **0 on `hosts`**, with `FLOW_ID`, `SRC_IP`, `DST_IP`,
`EVENT_TIME` and `LABEL` all absent. ID-typed columns are keys and edges, never
features.

Nothing else is dropped: every remaining column is a per-flow traffic statistic
computed from that flow's own packets.

**One residual-leakage question is left open.** Dropping `capture_id` removes
the column, but if hosts are largely capture-private then graph structure alone
reconstructs it — two flows sharing an endpoint would be strong evidence of
sharing a capture, and therefore of sharing a label. That is plausible here:
`hosts` is exactly the endpoint set (407,244, matching `COUNT(DISTINCT ip)` over
both foreign keys on the nose) and the testbed used a fixed victim range. It was
**not measured**, and it should be before any of these per-class numbers are
quoted as detection quality.

## 10. Carried over from rows 1–2, confirmed again

- `scikit-learn` missing from the fat wheel → `Optuna search analysis failed` on
  every run. Non-fatal; fixed upstream in
  [kumo-ai/kumo#30507](https://github.com/kumo-ai/kumo/pull/30507).
- `--no-publish-snapshots` (`kumo.NoopArtifactLocation`) used for both tiers.
- `SnowflakeConnector(warehouse=...)` inert locally; warehouse set in the
  session factory.
- Tables not stages, so one typed `NETWORK_FLOWS_RAW` copy is built first and
  every tier is a nested deterministic hash sample off it.
- Parts are **capture-aligned** (as row 2, unlike row 1): `part-00003` is one
  session, 100% Benign. A prefix slice would contain no attacks at all — hence
  hash sampling.
- `validation` metrics come back empty in the SDK's metric grouping on all three
  fixtures.

## Reproducing

```bash
source scripts/_env.sh  # resolves the monorepo SDK and root .venv

"$KUMO_SDK_PYTHON" scripts/kaggle/bccc/probe_bccc_fixture.py
"$KUMO_SDK_PYTHON" scripts/kaggle/bccc/build_bccc_tables.py --rebuild-raw --raw-only
scripts/kaggle/bccc/run_bccc_tier.sh small
```

The raw build is ~25 min on KUMO_L and only needs doing once; every tier derives
from it. Snowflake objects (account TLB22710): stage
`HYU.KAGGLE_FIXTURES.BCCC_IDS_20GB`; tables `HYU.BCCC.NETWORK_FLOWS_RAW` plus
`NETWORK_FLOWS_<TIER>`, `NETWORK_FLOWS_<TIER>_TRUTH`, `HOSTS_<TIER>`.

## Scripts

| file                    | purpose                                                            |
| ----------------------- | ------------------------------------------------------------------ |
| `probe_bccc_fixture.py` | stage the S3 fixture; manifest and inferred schemas                |
| `probe_bccc_stats.py`   | every matrix claim checked: keys, classes, leakage, per-part shape |
| `build_bccc_tables.py`  | typed raw copy, then nested hash-sampled tiers                     |
| `run_bccc_local.py`     | the end-to-end run, with multiclass held-out scoring               |
| `run_bccc_tier.sh`      | build + run one tier unattended, with disk sampling                |
