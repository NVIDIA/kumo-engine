# `kaggle_iot23_flow_detailed_label_10gb` on the SDK's local backend

Row 4 of [`kaggle-fixture-matrix.md`](../../../kaggle-fixture-matrix.md), run end
to end in-process, the same way [row 1](../ethereum/FINDINGS.md),
[row 2](../iot23/FINDINGS.md) and [row 3](../bccc/FINDINGS.md) were.

This is **row 2's fixture with the target swapped**: same S3 root, same stage,
same tables, same graph, same sample predicate, same holdout predicate — and,
verified below, exactly the same rows in every tier. Only
`is_malicious` → `detailed_label` differs. That makes it the one row in the
matrix where a difference in the result is attributable to the *task* rather
than to the data, and §9 is the comparison that follows.

**Result: the pipeline works. Three tiers complete end to end, uncontended.**
But two things have to be said before any number is quoted. `test_acc` is not
merely uninformative here, it is **actively misleading** — 0.1703 at the largest
tier against a test split whose majority class is 88.8% (§5). And the leakage
warning inherited from row 2 turns out **not to transfer**: the "which capture
is this?" channel that is worth 0.9256 AUROC on the binary target is worth
0.7704 accuracy and **0.2665 macro-F1** on this one (§4).

## Runs

|                               |            smoke |               small |               **part** |
| ----------------------------- | ---------------: | ------------------: | ---------------------: |
| flows (labeled + held out)    |          163,106 |           3,254,288 |         **16,266,495** |
| hosts                         |          160,187 |           3,184,500 |             15,742,423 |
| labeled / held out            | 130,735 / 32,371 | 2,604,223 / 650,065 | 13,013,748 / 3,252,747 |
| classes present (of 16)       |            **8** |              **10** |                 **12** |
| classes with held-out support |                6 |                   9 |                     10 |
| train                         |          159.1 s |           2,598.2 s |          **3,721.1 s** |
| predict                       |            2.4 s |              25.3 s |                156.3 s |
| DiskGraph index               |           154 MB |              1.6 GB |                 6.1 GB |
| predictions parquet (long)    |           1.5 MB |               38 MB |                 198 MB |
| peak `/mnt/localssd`          |         0.21 GiB |            2.27 GiB |               8.78 GiB |
| `test_acc` (reported)         |           0.8942 |              0.7826 |             **0.1703** |
| `test_refit_acc`              |           0.3429 |              0.7831 |                 0.2344 |
| — test-split majority rate    |           0.8882 |              0.8886 |                 0.8883 |
| **held-out accuracy**         |           0.6837 |          **0.9778** |                 0.9237 |
| **held-out macro-F1**         |   0.3208 (6 cls) |  **0.7048 (9 cls)** |        0.5253 (10 cls) |
| — held-out majority rate      |           0.6569 |              0.6563 |                 0.6578 |
| — capture-only baseline (§4)  |                — |     0.7700 / 0.3010 |        0.7704 / 0.2665 |

Every run was uncontended: nothing else occupied the 4 vCPUs for the duration of
any of the three, so unlike rows 2 and 3 these timings are measurements rather
than upper bounds. Held-out figures are measured against labels blanked out of
the graph entirely (§1), on a uniform hash sample the model never saw — 100% of
the held-out rows at `smoke` and `small`, 20% (651,827 rows) at `part`.

**Read the held-out columns, not `test_acc`.** See §5.

## 1. The static-target holdout, fourth confirmed instance — carried over

Exactly as [row 1 §1](../ethereum/FINDINGS.md), [row 2 §1](../iot23/FINDINGS.md)
and [row 3 §2](../bccc/FINDINGS.md). `PREDICT network_flows.detailed_label` is a
static column target, so Kumo compiles the prediction table to
`WHERE detailed_label IS NULL`; the fixture labels all 325,309,946 flows, so
batch prediction as published selects nothing.

Row 1 §1 named rows 2, 3, 4 and 10 as the affected set. **This is the fourth,
and it closes the classification rows** — only row 10
(`yambda.played_ratio_pct`) is left to confirm. `--holdout-pct 20` blanks a
deterministic slice and keeps the true labels in a `*_TRUTH` side table.

One wrinkle this row adds: **`'-'` is a class, not a missing value.** Benign
flows carry the literal string `'-'` in `detailed_label`, so after the holdout
the column has three meanings — a family name, `'-'` for benign, and SQL `NULL`
for "held out". Blanking has to write `NULL` and nothing else, and the scoring
side has to treat `'-'` as an ordinary class. It is a 9.5% class, so getting it
wrong would not be subtle, but it is the first fixture where the benign case is
spelled rather than absent.

## 2. Row 2's raw table does not carry the target, and re-reading two columns costs as much as twenty-one — NEW

`HYU.IOT23.NETWORK_FLOWS_RAW` holds all 325,309,946 flows typed, and it was
reused unchanged. But it has no `detailed_label` column at all: row 2's build
dropped the column in the `SELECT` that created the table, because for a binary
target it is a leak.

So the target had to be read out of the stage again. `DETAILED_LABELS_RAW` is
that read, two columns wide (`flow_id`, `detailed_label`) instead of
twenty-one — the expectation being that Snowflake would prune the Parquet and
make it much cheaper than row 2's full pass.

**It did not.** Measured on `KUMO_L`:

|                                                                                 |             |
| ------------------------------------------------------------------------------- | ----------: |
| `DETAILED_LABELS_RAW` (2 columns, 325,309,946 rows, 16 distinct labels, 0 null) | **726.2 s** |
| row 2's whole-stage scans, 21 columns (measured range)                          |  600–1300 s |

A two-column projection lands inside the range of the twenty-one-column ones.
The stage is read through `$1:<field>` on a Parquet external stage and the
variant decode evidently dominates, not the number of fields kept.

**The lesson for every later row: materialize every column you might ever want
on the first pass.** A second pass for one more column costs essentially a first
pass, so row 2 dropping `detailed_label` at ingest saved nothing and cost row 4
twelve minutes. Note this is the opposite of the intuition that a narrow
projection is cheap — it is cheap on a *table*, not on a variant-typed stage.

Everything after that is cheap, because every tier is a join of two native
Snowflake tables:

| tier  | flows table | truth table | hosts table |
| ----- | ----------: | ----------: | ----------: |
| smoke |       6.6 s |       1.7 s |       1.0 s |
| part  |      10.6 s |       1.6 s |       1.3 s |

## 3. The leak inverts, and one half of it is exact — NEW

Row 2 dropped `label`, `detailed_label`, `scenario_id`, `uid` and `ts`. For row
4, `detailed_label` is the target and two of the others change sides.

### `is_malicious` is an exact collapse of the target

Row 2's *target* is row 4's *leak*, and it is perfect:

|                                                      |             measured |
| ---------------------------------------------------- | -------------------: |
| rows where `is_malicious <> (detailed_label <> '-')` | **0 of 325,309,946** |

Not "close to" — zero exceptions over the whole fixture. Keeping the column
would hand the model a free reduction of 16 classes to 2, and would give away
the 9.5% benign class outright. It is excluded from every tier table in the
build rather than merely left out of the run's column list, so a careless
`SELECT *` cannot reintroduce it. `label` (`'Malicious'`/`'Benign'`) is the same
collapse as a string and never entered `NETWORK_FLOWS_RAW` at all.

### There is a 16th class the usual class listings omit

Measured over all 325,309,946 rows:

| class                              |        rows |  share |
| ---------------------------------- | ----------: | -----: |
| `PartOfAHorizontalPortScan`        | 213,852,924 | 65.74% |
| `Okiru`                            |  60,990,708 | 18.75% |
| `-` (benign)                       |  30,860,691 |  9.49% |
| `DDoS`                             |  19,538,713 |  6.01% |
| `C&C-HeartBeat`                    |      33,673 | 0.010% |
| `C&C`                              |      21,995 | 0.007% |
| `Attack`                           |       9,398 | 0.003% |
| `C&C-PartOfAHorizontalPortScan`    |         888 |        |
| `C&C-HeartBeat-Attack`             |         834 |        |
| `C&C-FileDownload`                 |          53 |        |
| `C&C-Torii`                        |          30 |        |
| `FileDownload`                     |          18 |        |
| **`C&C-HeartBeat-FileDownload`**   |      **11** |        |
| `PartOfAHorizontalPortScan-Attack` |           5 |        |
| `Okiru-Attack`                     |           3 |        |
| `C&C-Mirai`                        |           2 |        |

`C&C-HeartBeat-FileDownload` (11 rows) is the sixteenth and is missing from the
usual 15-row listing. The four head classes are 99.98% of the fixture; the
twelve tail classes total 11,237 rows.

**No tier is the published 16-class problem**, and the tiers are not each
other's problem either: 8 classes at `smoke`, 10 at `small`, 12 at `part`. Any
cross-tier comparison compares different tasks (row 3 §1, confirmed here).

## 4. The capture leak is real but *far weaker* than for the binary target — NEW, and it corrects the prior

This is the finding worth the most, because it inverts what row 2 §3 sets up.

Row 2 measured that dropping `scenario_id` does not remove the capture: 99.87%
of hosts appear in exactly one capture, both endpoint IPs are edges into
`hosts`, so graph structure reconstructs it — and for the binary target that
reconstruction is worth **0.9256 AUROC with no model at all**. The natural
inference, and the one stated going into this row, was that it must be *worse*
here, because a capture is one malware sample and row 4's classes are malware
families.

**Measured, it is not.** The mechanism is identical — re-measured on this row's
tiers, and matching row 2 as it must, since the rows are the same:

|                              |                `small` |                  `part` |
| ---------------------------- | ---------------------: | ----------------------: |
| hosts                        |              3,184,500 |              15,742,423 |
| hosts in exactly one capture | 3,180,323 (**99.87%**) | 15,643,208 (**99.37%**) |
| mean captures per host       |                 1.0013 |                  1.0064 |

But what the capture *tells you* is much less. IoT-23's captures are **not
single-family**, unlike BCCC's 30 sessions, which row 3 §9 measured at purity
1.000000. Over all 325,309,946 rows, 23 captures:

| statistic                         |                                                                value |
| --------------------------------- | -------------------------------------------------------------------: |
| captures with purity 1.000        | **3 of 23** — and all three are honeypots of 452, 1,374 and 130 rows |
| lowest purity                     |   **0.4997** (`CTU-IoT-Malware-Capture-17-1`, 54.7M rows, 7 classes) |
| the five largest captures' purity |                               0.4997, 0.5631, 0.7246, 0.9999, 0.9999 |
| classes per capture               |                                                               1 to 7 |

And the inverse direction, which decides whether recovering the capture hands
you a class at all:

| class                       |        rows | captures it appears in | share in its largest capture |
| --------------------------- | ----------: | ---------------------: | ---------------------------: |
| `PartOfAHorizontalPortScan` | 213,852,924 |                     11 |                    **0.344** |
| `Okiru`                     |  60,990,708 |                      5 |                    **0.224** |
| `-`                         |  30,860,691 |                     23 |                        0.667 |
| `DDoS`                      |  19,538,713 |                      7 |                        0.699 |
| `C&C-HeartBeat`             |      33,673 |                      5 |                        0.466 |
| `C&C`                       |      21,995 |                     10 |                        0.374 |

Only classes at 888 rows and below are single-capture, and those are noise at
any tier that contains them at all.

So the no-model ceiling is far lower than row 2's. Predicting, for every flow,
the majority class of its own capture (fitted on training rows only, so it never
sees a held-out label):

|                     | whole fixture |   `small` held-out |     `part` held-out |
| ------------------- | ------------: | -----------------: | ------------------: |
| accuracy            |    **0.7706** |         **0.7700** |          **0.7704** |
| macro-F1            |             — | **0.3010** (9 cls) | **0.2665** (10 cls) |
| majority-class rate |        0.6574 |             0.6563 |              0.6578 |

**The honest statement is the opposite of the one row 2 §3 sets up.** For
`is_malicious`, "which capture is this?" is nearly the whole task — 0.9256 AUROC
on a 90.5% base rate. For `detailed_label` it is worth about eleven points of
accuracy over a constant predictor and almost nothing in macro-F1, because the
big captures are genuine mixtures of scanning, Okiru, DDoS and benign traffic.
**Row 2 §3's warning does not transfer to row 4 at the strength it was stated.**

It still has to be quoted beside the results — 0.77 accuracy / 0.27–0.30 macro-F1
is the floor, not zero, and `small`'s 0.9778/0.7048 should be read against it —
but this task is *not* substantially "identify the capture", and the model's
margin over the structural baseline is where its value is. `part`'s 0.9237
accuracy is +15 points over the baseline; `small`'s 0.9778 is +21. `smoke`'s
0.6837 is **below** it, which is the clearest statement available that the smoke
tier is too small to mean anything.

## 5. `test_acc` is worse than uninformative — NEW, and it degrades with scale

Row 3 §3 found `test_acc` landing exactly on the test split's majority rate.
Here it lands far *under* it, and the gap widens with every tier:

| tier  | reported `test_acc` | test-split majority | reported minus majority |
| ----- | ------------------: | ------------------: | ----------------------: |
| smoke |              0.8942 |              0.8882 |                  +0.006 |
| small |              0.7826 |              0.8886 |              **−0.106** |
| part  |          **0.1703** |              0.8883 |              **−0.718** |

A model that always answered `PartOfAHorizontalPortScan` would score 0.888 on
the test split at every tier. The `part` model scores 0.170 there while scoring
0.9237 accuracy and 1.0000 F1 on `Okiru` against held-out ground truth. Both are
true; they are measured on different distributions.

The cause is the split, measured from `published-model/manifests/input.json`
(`stage_counts`/`stage_seed_times`) with per-class counts recovered from
Snowflake over each stage's window. `describe_split` runs **before** the score
check, deliberately, because rows 2 and 3 both diagnosed this late.

`part` tier:

| stage | span                    |       rows | classes | composition                                          |
| ----- | ----------------------- | ---------: | ------: | ---------------------------------------------------- |
| train | 246.15 days             | 10,409,622 |       9 | PoAHPS 64.10%, Okiru 21.76%, `-` 8.00%, DDoS 6.12%   |
| val   | **0.1105 days (2.7 h)** |  1,316,676 |       6 | PoAHPS 56.23%, `-` 30.56%, Okiru 13.01%              |
| test  | 204.18 days             |  1,286,578 |      10 | **PoAHPS 88.83%, DDoS 11.11%** — 99.94% between them |

`small` and `smoke` are the same shape: test is 88.86%/88.82% PoAHPS plus
11.08%/11.13% DDoS, and the validation window is 1.7 hours and 28 minutes wide
respectively.

Three consequences:

- **`Okiru` is absent from test entirely at all three tiers** — 21.8% of train,
  18.8% of the fixture, zero rows in the test stage. A metric that cannot see
  the second-largest class is not a statement about the task.
- **Train and test are different distributions**, not two samples of one: train
  is 64% PoAHPS, test is 89% PoAHPS with Okiru removed. A model calibrated on
  the first is *penalised* on the second, and the penalty grows with how well it
  fits train — which is exactly the monotone degradation in the table above.
  `test_acc` here measures distribution shift, not model quality.
- **Model selection is near-blind.** Validation is a 2.7-hour window whose class
  mix matches neither train nor test, and `val_acc` reads 0.993–0.998 across
  every Optuna trial, so it barely discriminates.

The pathology is structural, not a sample-size effect: the split is cut by row
*quantile* while row density over time is wildly uneven, and captures are
time-contiguous. It will not improve with a bigger tier — it gets worse.

**Do not quote `test_acc` or `test_refit_acc` for this fixture.** The held-out
score is the one that means anything, and it should be quoted beside the §4
baseline.

## 6. Held-out quality is not monotone in scale — NEW

| tier  | held-out accuracy | held-out macro-F1 | classes with support |
| ----- | ----------------: | ----------------: | -------------------: |
| smoke |            0.6837 |            0.3208 |                    6 |
| small |        **0.9778** |        **0.7048** |                    9 |
| part  |            0.9237 |            0.5253 |                   10 |

`small` is the best run on both metrics, and `part` — 5× the data — is worse.
The per-class tables say why, and it is not noise:

- `Okiru` is **perfect at both** — precision 1.0000, recall 1.0000, on 122K and
  122K held-out rows. Nothing regressed there.
- The regression is **precision on `-` and `C&C`**. At `part` the model predicts
  `-` for 108,154 rows against a true support of 61,869 (precision 0.5710,
  recall 0.9982), and `C&C` for 1,111 rows against a support of 52 (precision
  0.0432, recall 0.9231). At `small` the same two are 0.8986 and 0.9714.
- So the larger model **over-predicts the minority classes**: it buys recall on
  the tail at a large cost in precision, and macro-F1 punishes that.

Part of this is attributable to §7 — the run tunes on `acc` computed on a test
split that contains almost none of `-` (0.04%) and none of `Okiru`, so the
selection signal for over-predicting `-` is absent. It is stated here as a
measured non-monotonicity with a visible mechanism, not as a diagnosis: a
fourth tier would be needed to say whether `small` is a lucky draw or `part` is
a real regression.

The comparison is at least clean — the tiers are nested samples of the same
fixture with the same holdout, so `small`'s held-out rows are a subset of
`part`'s and the two models are scored on overlapping ground truth.

## 7. Model-plan overrides are still silently dropped — carried over from row 3 §7, reproduced on a second fixture

Row 3 measured this and could not isolate the drop point. This run reads the
resolved plan back on every tier and prints the verdict, and it reproduces
exactly, at all three:

```
metrics      asked=['acc', 'precision', 'recall', 'f1', 'mrr'] resolved=['acc'] -> DROPPED
tune_metric  asked='f1' resolved='acc'                                          -> DROPPED
```

So the defect is **not BCCC-specific** — same result on a 21-column fixture with
a different graph, which removes "something about that fixture" as an
explanation and leaves it in `kumo_core`'s plan resolution.

It matters more here than in row 3, and for a sharper reason than "accuracy is a
bad metric on a long tail". The metric it falls back to is computed on the test
split of §5, which holds two classes of the twelve present. Tuning on `acc`
therefore means tuning on a measurement of distribution shift.

## 8. `_column_stats` corruption does not bite this fixture — carried over from row 3 §6, checked rather than assumed

Row 3 found that `kumo_core/pquery.py::_column_stats` matches `ColStatType`
tokens by substring against the whole key, column name included, so a column
whose *name* contains `MEAN`, `MIN`, `MAX`, `COUNT`, `STD`, … loses its real
statistics. 142 of 323 BCCC columns were affected.

`run_iot23_dl_local.py::check_stat_token_collisions` simulates the buggy
resolution — for each stat token, form the key the writer would produce and ask
which token the reader would match — and prints the answer on every run:

```
no stat-token collisions in 19 column names (row 3 S6 does not bite this fixture)
```

The check was exercised against known-bad names as a control and reproduces row
3's cases exactly:

```
PAYLOAD_BYTES_MEAN: ['MIN->MEAN', 'MAX->MEAN', 'STD->MEAN']
PACKETS_COUNT:      ['MEAN->COUNT', 'MIN->COUNT', 'MAX->COUNT', 'STD->COUNT']
MAX_HEADER_BYTES:   ['STD->MAX']
```

So the defect is confirmed present in the installed `kumo_core` and confirmed
inert for row 4's 19 columns. This is a check to run on any new fixture, not a
fact to inherit.

## 9. What the target swap actually cost — the row-2 comparison

Every tier here holds **exactly the same flows** as row 2's tier of the same
name, with exactly the same rows held out: the sample predicate
`MOD(ABS(HASH(flow_id)), 10000) < N` and the holdout predicate
`MOD(ABS(HASH(flow_id || '/holdout')), 100) < 20` are byte-identical. Confirmed
by the row counts, which match to the row:

| tier  |      flows |      hosts |  held out | row 2     |
| ----- | ---------: | ---------: | --------: | --------- |
| smoke |    163,106 |    160,187 |    32,371 | identical |
| small |  3,254,288 |  3,184,500 |   650,065 | identical |
| part  | 16,266,495 | 15,742,423 | 3,252,747 | identical |

### What did not change

- **The graph, exactly.** Two tables, two foreign keys into one destination,
  `hosts` one column wide and featureless. Same edges, same pinned stypes, same
  four inference corrections (`SOURCE_IP` categorical→ID, `DESTINATION_IP`
  text→ID, `DESTINATION_PORT` and `SOURCE_BYTES` categorical→numerical).
- **The index size, nearly.** 154 MB / 1.6 GB / 6.1 GB against row 2's
  0.21 / 1.77 / 7.52 GiB of *peak* footprint. The graph is the same, so the
  index is the same up to one column's width.
- **The build machinery**, minus the target column: the tier build is a
  near-copy of row 2's, and the only structural change is a join.
- **Every defect.** Static target (§1), missing `sklearn`, dropped model plan,
  empty `validation` metric group, capture-aligned parts.

### What did change

|                               | row 2 (`is_malicious`)                   | row 4 (`detailed_label`)                              |
| ----------------------------- | ---------------------------------------- | ----------------------------------------------------- |
| task                          | binary                                   | 16-class (8/10/12 per tier)                           |
| metrics available             | AUROC, AP, acc                           | acc, precision, recall, f1, mrr — **no AUROC, no AP** |
| prediction shape              | wide, `True_PROB` per entity             | **long**, entity × class                              |
| prediction rows at `part`     | 3.25M                                    | **39,032,964** (3.25M × 12)                           |
| predictions parquet at `part` | —                                        | 198 MB                                                |
| capture-only baseline         | **0.9256 AUROC**                         | **0.7704 acc / 0.2665 macro-F1**                      |
| `test_acc` vs test majority   | skewed but on-trend                      | **−0.72 at `part`** (§5)                              |
| leaks dropped                 | `label`, `detailed_label`, `scenario_id` | `is_malicious`, `label`, `scenario_id`                |

### The multiclass premium in training time, measured on identical rows

| tier  | row 2 train | row 4 train |                      ratio |
| ----- | ----------: | ----------: | -------------------------: |
| smoke |     132.9 s |     159.1 s |                  **1.20×** |
| small |   1,801.2 s |   2,598.2 s |                  **1.44×** |
| part  | 4,454.5 s † |   3,721.1 s | 0.84× — **not comparable** |

† Row 2's `part` was contended for its whole duration and is an upper bound;
row 4's was not. The honest reading is that a 16-way softmax over the same graph
costs **20–45% more** than a binary one at the tiers where both are clean, and
that the `part` row cannot be used to extend that.

### Scaling within row 4, and what it does not support

| tier  |      flows |     train | ×rows | ×time | implied exponent |
| ----- | ---------: | --------: | ----: | ----: | ---------------: |
| smoke |    163,106 |   159.1 s |     — |     — |                — |
| small |  3,254,288 | 2,598.2 s | 20.0× | 16.3× |             0.93 |
| part  | 16,266,495 | 3,721.1 s |  5.0× | 1.43× |         **0.22** |

The two exponents differ by a factor of four, so **no extrapolation to full
scale is defensible from these three points.** Carrying 0.22 forward gives ~2 h
for 325M rows; carrying 0.93 forward gives ~17 h; linear gives ~21 h. All three
are consistent with the data and the spread is the finding. Row 2 reached the
same impasse from its own three points (its §5 retracted an earlier estimate for
the same reason), so this appears to be a property of how AutoML cost moves with
this fixture rather than of either target.

Disk is the tractable half. The `part` index is 6.1 GB for 16.27M flows, or
**0.375 GB per million rows**, which puts 325M rows at **~122 GB** against
217 GB free — feasible, with room for the ingest transient. Note the long-format
predictions are a real second cost at scale: 198 MB for 3.25M held-out entities
× 12 classes, so a full-scale 20% holdout over 16 classes is roughly 5 GB of
predictions alone.

## 10. Carried over from rows 1–3, confirmed again

- `scikit-learn` missing from the fat wheel → `Optuna search analysis failed` /
  `ModuleNotFoundError: No module named 'sklearn'` on every run. Non-fatal;
  fixed upstream in [kumo-ai/kumo#30507](https://github.com/kumo-ai/kumo/pull/30507).
- **Multiclass plumbing** exactly as row 3 §5: no AUROC/AP; predictions long
  (`ENTITY | PREDICTED | CLASS | SCORE`, `PREDICTED` marking the argmax);
  `num_entities_predicted` reports **rows, not entities** — 258,968 for 32,371
  entities × 8 classes at smoke, 39,032,964 for 3,252,747 × 12 at part. Top-20
  truncation never bites: the widest tier has 12 classes.
- `validation` metrics come back empty in the SDK's metric grouping, on all four
  fixtures now.
- `--no-publish-snapshots` (`kumo.NoopArtifactLocation`) used for every tier.
- `SnowflakeConnector(warehouse=...)` inert locally; the warehouse is set in the
  session factory.
- `Table.infer_metadata` discards stypes set before it, so `PINNED_STYPES` is
  applied afterwards.
- Parts are capture-aligned, so tiers are hash samples of `flow_id` rather than
  prefixes; a prefix here is one malware family.
- `event_time` is microseconds; data spans 2018-05-09 to 2019-09-21.
- **The in-tree SDK on `PYTHONPATH` is not optional.** Launching a run without
  it fails in the first second with `AttributeError: module 'kumoai_studio' has no attribute 'NoopArtifactLocation'`, because `kumoai_studio` resolves to the fat wheel
  instead of the working tree. Cheap to diagnose, but silent if the launcher
  swallows stderr.

## Reproducing

```bash
source scripts/_env.sh  # resolves the monorepo SDK and root .venv

# One-time, ~12 min: the two-column target table beside row 2's raw copy.
"$KUMO_SDK_PYTHON" scripts/kaggle/iot23_detailed/build_iot23_dl_tables.py \
    --labels-only

# Build one tier and run it, sampling free disk throughout.
# Scales: smoke | small | part | half | full (parts per 10,000 of flow_id).
scripts/kaggle/iot23_detailed/run_iot23_dl_tier.sh part 20

# What the score actually measures (S4).
"$KUMO_SDK_PYTHON" scripts/kaggle/iot23_detailed/probe_iot23_dl_leakage.py --raw
"$KUMO_SDK_PYTHON" scripts/kaggle/iot23_detailed/probe_iot23_dl_leakage.py \
    --scale part
```

Snowflake objects (account `TLB22710`): the stage
`HYU.KAGGLE_FIXTURES.IOT23_10GB` and the table `HYU.IOT23.NETWORK_FLOWS_RAW`
are **row 2's, reused unchanged and never rebuilt**. This row adds
`HYU.IOT23.DETAILED_LABELS_RAW` (325,309,946 rows, two columns) plus
`NETWORK_FLOWS_DL_<TIER>`, `NETWORK_FLOWS_DL_<TIER>_TRUTH` and
`HOSTS_DL_<TIER>` per tier. Row 2's `NETWORK_FLOWS_<TIER>` tables are left
alone — they dropped the target and are unusable here.

## Scripts

| file                        | purpose                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| `build_iot23_dl_tables.py`  | the two-column target table, then hash-sampled tiers joined off row 2's raw copy                 |
| `run_iot23_dl_local.py`     | the end-to-end run: multiclass scoring, split description, resolved-plan check, stat-token check |
| `probe_iot23_dl_leakage.py` | what the score measures: capture purity, class concentration, the no-model baseline              |
| `run_iot23_dl_tier.sh`      | build + run one tier unattended, with disk sampling                                              |
