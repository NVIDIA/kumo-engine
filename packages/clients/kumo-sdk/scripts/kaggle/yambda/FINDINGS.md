# `kaggle_yambda_*_100gb` on the SDK's local backend

Rows **6, 7, 10, 13 and 15** of [`kaggle-fixture-matrix.md`](../../../kaggle-fixture-matrix.md)
— one fixture, one graph contract, five task families — run the same way rows
1–5, 8, 9, 11, 12, 14 and 16 were. This block closes the matrix.

```
row 6   PREDICT LIST_DISTINCT(interactions.event_type)
        CLASSIFY TOP 5    FOR EACH users.user_id
row 7   PREDICT LIST_DISTINCT(interactions.event_type, 0, 30, days)
        RANK TOP 5        FOR EACH users.user_id
row 10  PREDICT interactions.played_ratio_pct
        FOR EACH interactions.interaction_id
row 13  PREDICT LIST_DISTINCT(interactions.item_id)
        RANK TOP 100      FOR EACH users.user_id
row 15  PREDICT LIST_DISTINCT(interactions.item_id, 0, 30, days)
        RANK TOP 100      FOR EACH users.user_id
```

Verdict: **three of five run as published**; **row 6 is not expressible at all**
and **row 13 is not expressible against the graph the matrix's own contract
specifies**. Where a model ran, it lost to a one-line `GROUP BY` on three of the
four scored rows, and on the fourth (row 10) it did not beat a constant.

| headline                                               | measured                                                                                                                                 |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **the fixture is real and complete**                   | 101 files, **109,221,203,314 B**, **4,796,250,510 rows**, valid manifest — only the *matrix* had placeholders (§1)                       |
| **row 6 cannot compile under any graph**               | it misses *both* axes of `time_range_validator`'s 2×2 at once; Yambda is the first fixture to fill all four cells (§2)                   |
| **rows 13 and 15 need different graphs**               | with `EVENT_TIME`: 13 rejected. Without it: 15 rejected. The contract says `event_time` *is* the event time (§2)                         |
| **the anchor edge is not a trap here**                 | both foreign keys resolve identically because they point at *different* tables — the first fixture in the series where this is true (§3) |
| **row 10 is a constant predictor**                     | R² **0.0007**, prediction sd **7.70** against truth sd **44.10**; `GROUP BY user_id` halves the MAE (§5)                                 |
| **CLASSIFY vs RANK: no measurable difference**         | MAP@1 0.9923 vs 0.9923 at `y_smoke`; the AUROC gap is inside the self-noise band (§6)                                                    |
| **row 15's entire score is re-listening**              | unfiltered MAP@1 0.2717 → filtered **0.0110** (§7)                                                                                       |
| **row 13 loses to a co-occurrence `GROUP BY` by 2.4×** | at both tiers, filtered MAP@100 0.0275 vs 0.0673 (§7)                                                                                    |
| **noise floor is 6%, not BCCC's 38%**                  | two runs on byte-identical data: MAP@100 0.0538 / 0.0569 (§8)                                                                            |
| **disk would not hold the full index**                 | 50.9 B/interaction ⇒ **244 GB** against 217 GB free — a first in the series (§9)                                                         |

## Runs

4 vCPU, 15 GB RAM, one L4, `/mnt/localssd` empty at the start, nothing else
running — timings are measurements, not upper bounds.

|                      |  r10 `y_tiny` | r6\* `y_tiny` | r7 `y_tiny` |    r7 B | r15 `y_tiny` |   r15 B | r13 `y_tiny` | r15 `y_smoke` | r13 `y_smoke` | r7 `y_smoke` | r6\* `y_smoke` |
| -------------------- | ------------: | ------------: | ----------: | ------: | -----------: | ------: | -----------: | ------------: | ------------: | -----------: | -------------: |
| end to end           |       **yes** |       **yes** |     **yes** | **yes** |      **yes** | **yes** |      **yes** |       **yes** |       **yes** |      **yes** |        **yes** |
| users                |           212 |           212 |         212 |     212 |          212 |     212 |          212 |         1,079 |         1,079 |        1,079 |          1,079 |
| interactions         |       984,904 |       984,904 |     984,904 | 984,904 |      984,904 | 984,904 |      833,296 |     5,097,595 |     4,151,457 |    5,097,595 |      5,097,595 |
| candidate items      |             — |             — |           — |       — |      117,378 | 117,378 |      117,378 |       296,981 |       296,981 |            — |              — |
| training-table rows  |       762,395 |        10,600 |      10,600 |  10,600 |       10,600 |  10,600 |          212 |        53,950 |         1,079 |       53,950 |         53,950 |
| train                | **2,115.4 s** |       217.3 s |     239.5 s | 237.8 s |  **166.0 s** | 160.6 s |  **143.5 s** |   **453.6 s** |       242.9 s |  **480.2 s** |        326.7 s |
| predict              |         9.6 s |         0.7 s |       0.5 s |   0.6 s |        0.9 s |   0.9 s |        0.7 s |         1.4 s |         1.1 s |        0.7 s |          0.7 s |
| DiskGraph index      |       49.8 MB |       50.7 MB |     50.7 MB |       — |      50.7 MB |       — |      32.6 MB |  **259.7 MB** |      161.9 MB |     259.7 MB |              — |
| peak RSS             |   **4.00 GB** |       5.08 GB |     1.96 GB | 1.96 GB |      2.05 GB | 2.02 GB |      2.27 GB |   **2.75 GB** |       2.72 GB |      1.97 GB |        1.97 GB |
| peak `/mnt/localssd` |       1.29 GB |       1.30 GB |     1.20 GB |       — |      1.80 GB | 2.58 GB |      2.38 GB |       5.21 GB |   **6.97 GB** |      7.03 GB |        7.12 GB |

`r6*` is **row 6 repaired** — the published row 6 does not compile (§2). `B`
columns are byte-identical repeats for the noise floor (§8).

Deliberate failures, both reproduced on the real graph, both **before ingest**:

|                          |                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------- |
| `--row 6` (as published) | **rejected** — *"static usage of LIST_DISTINCT is only possible for RANK problems"*    |
| `--row 13 --time-column` | **rejected** — *"Time range required for LIST_DISTINCT with time column (EVENT_TIME)"* |

## 1. The fixture, checked against the matrix — and the matrix was the thing that was wrong

The matrix published `YAMBDA_STORED_SIZE` and `YAMBDA_TOTAL_ROWS` **literally**,
in all five Yambda rows, where every other fixture carried real numbers. That is
a strong signal a fixture was templated and never built, so nothing was planned
until the prefix had been listed through a Snowflake external stage.

**It is there, and it is complete.** `_metadata/manifest.json` reads:

|                |                                                              manifest |
| -------------- | --------------------------------------------------------------------: |
| `interactions` |                   96 files, 109,192,737,220 B, **4,785,859,887 rows** |
| `items`        |                              1 file, 25,739,789 B, **9,390,623 rows** |
| `users`        |                               1 file, 2,726,305 B, **1,000,000 rows** |
| **total**      | 101 files, **109,221,203,314 B (101.72 GiB)**, **4,796,250,510 rows** |

plus `LICENSE.txt` (Apache-2.0) and `ATTRIBUTION.md` (Yandex, Yambda-5B). A
bucket-wide `LIST ... PATTERN` for `/yambda/i` returns exactly these 101 objects
and nothing stray. **The placeholders are a documentation defect only** — they
have been filled in from the manifest (uncommitted).

This is by a wide margin the largest fixture in the matrix: 1.65× the rows of
NOAA's 2.9 billion and 5.4× the stored bytes of BCCC's 19.9 GB.

### Schema, read off the stage rather than off the matrix

| column                 | staged type    | note                                                                   |
| ---------------------- | -------------- | ---------------------------------------------------------------------- |
| `interaction_id`       | `NUMBER(38,0)` | globally sequential 0 … 4,785,859,886; unique                          |
| `user_id`              | `NUMBER(38,0)` | 1 … 1,000,000                                                          |
| `item_id`              | `NUMBER(38,0)` | 1 … 9,390,623                                                          |
| `event_time`           | `NUMBER(38,0)` | **no epoch**, values 50,000 … 130,000,000,000, all multiples of 25,000 |
| `event_type`           | `TEXT`         | **five values only**                                                   |
| `is_organic`           | `NUMBER(38,0)` | 0/1                                                                    |
| `played_ratio_pct`     | `NUMBER(38,0)` | **NULL on every non-`listen` row**, max **159**                        |
| `track_length_seconds` | `NUMBER(38,0)` | NULL on the same rows                                                  |

`users` and `items` are **one column wide** — bare id lists. That is load-bearing
for rows 13 and 15, where `items` is the RHS candidate table, and it was
confirmed from the trained model rather than assumed.

### `event_time` has no published unit, and the unit had to be inferred

Within one user's consecutive rows the gaps are 550,000 / 700,000 / 750,000 /
800,000 / 1,150,000 units, against `track_length_seconds × played_ratio_pct` of
107 / 145 / 160 / 230 seconds *actually played* on the row the gap starts from.
Under **milliseconds** every gap comfortably contains the audio it must contain.
Under any finer unit several gaps are **shorter than a track the row says was
played to 100%**, which is impossible.

At that unit the fixture spans **1,504 days**. Absolute calendar placement is
meaningless — Yambda's timestamps are shifted — but the *relative* spacing is
what a `0, 30, days` window depends on. **Stated as an inference, not a
measurement.** `event_time_raw` is carried on the raw table so the cast can be
revisited without re-reading the stage.

### The parquet is laid out in contiguous user blocks — and that changed the cost model

`part-00000` holds exactly **50,000,000 rows**, `user_id` **1 … 10,423,
contiguous and complete**. So each of the 96 parts is a block of ~10,420 *whole*
users.

Every previous block paid a full-stage scan because a hash predicate over
`user_id` cannot be pruned. Here four parts spread across the range were read
instead — **440.6 s** for 200,000,000 rows — and the eight boundary users were
dropped, since a truncated user is a *corrupted label* for four of the five rows.

The pool is **199,962,908 interactions, 41,975 users, 1,958,664 items**, and is
representative on the one checkable statistic: **4,764 interactions per user
against the fixture's 4,786**. Every tier is a hash sample of *this pool*.

### `played_ratio_pct` is NULL on exactly the non-`listen` rows — zero exceptions

|                                   | measured over the pool |
| --------------------------------- | ---------------------: |
| `played_ratio_pct IS NULL`        |          **5,777,036** |
| `event_type <> 'listen'`          |          **5,777,036** |
| `listen` rows that are null       |                  **0** |
| non-`listen` rows that are valued |                  **0** |

The two sets are *identical*. This is the most consequential fixture property in
the block, and §5 is about what it does to row 10.

| `event_type` |        rows |      share |  users |
| ------------ | ----------: | ---------: | -----: |
| `listen`     | 194,185,872 | **97.11%** | 38,722 |
| `like`       |   3,752,294 |      1.88% | 34,587 |
| `unlike`     |   1,391,260 |      0.70% | 26,785 |
| `dislike`    |     528,392 |      0.26% | 24,829 |
| `undislike`  |     105,090 |      0.05% | 12,067 |

**Five classes, and rows 6/7 ask for `TOP 5`.** The cut-off *is* the vocabulary.

**The fixture is not stationary.** Monthly volume grows 2.83M → 5.67M and the
active user base 18,694 → 35,770 across the span. Row 10's `TemporalSplit` cuts
by row quantile, so its test stage is a materially larger and later population
than its train stage.

## 2. What compiles — Yambda fills all four cells of the validator's 2×2 — NEW

`probe_yambda_query.py` compiles all five published queries against a synthesized
portable graph in about a second, with no data.

**With `EVENT_TIME` declared — the published graph contract:**

| row | verdict                                                                                |
| --- | -------------------------------------------------------------------------------------- |
| 6   | **rejected** — *"static usage of LIST_DISTINCT is only possible for RANK problems"*    |
| 7   | `multilabel_ranking`, `DateOffsetSplit(offsets=[-30,0], days)`                         |
| 10  | `regression`, `TemporalSplit([0.8, 0.1, 0.1])`                                         |
| 13  | **rejected** — *"Time range required for LIST_DISTINCT with time column (EVENT_TIME)"* |
| 15  | `temporal_link_prediction`, `DateOffsetSplit(offsets=[-30,0], days)`                   |

**With `EVENT_TIME` dropped:**

| row | verdict                                                                       |
| --- | ----------------------------------------------------------------------------- |
| 6   | **rejected** — *same message*; the objection is not the time column           |
| 7   | **rejected** — *"Missing a 'Create date' column in the table 'interactions'"* |
| 10  | `regression`, `RandomSplit([0.8, 0.1, 0.1])`                                  |
| 13  | `static_link_prediction`, `RandomLinkSplit([0.8, 0.1, 0.1])`                  |
| 15  | **rejected** — *"Missing a 'Create date' column"*                             |

**So rows 7 and 15 require the time column and row 13 requires its absence.** A
user who follows the published graph contract literally can run rows 7, 10 and 15
and **cannot run row 13**. BCCC row 14 §2 established this mechanism; Yambda is
where the matrix asks for *both halves against one root*, so it is where the
defect actually costs something.

### The 2×2, all four cells, on one fixture

`kumopql/validator/time_range_validator.py::_validate_aggr` allows an unwindowed
`LIST_DISTINCT` in exactly one cell. NOAA reached the non-foreign-key side; BCCC
row 14 reached the `CLASSIFY`-over-a-foreign-key side. Yambda has **both a
genuine foreign key (`item_id`) and a plain categorical (`event_type`) on one
fact table**:

| target                   | verb       | verdict                                                               |
| ------------------------ | ---------- | --------------------------------------------------------------------- |
| foreign key `ITEM_ID`    | `RANK`     | **legal** → `static_link_prediction` (row 13)                         |
| foreign key `ITEM_ID`    | `CLASSIFY` | rejected — *"only possible for RANK problems"*                        |
| categorical `EVENT_TYPE` | `RANK`     | rejected — *"only possible for the target clause with a foreign key"* |
| categorical `EVENT_TYPE` | `CLASSIFY` | rejected — *"only possible for RANK problems"*                        |

**Row 6 misses on both axes at once**, which is why no graph edit rescues it.

The smallest repair that compiles is the window row 7 already has:
`LIST_DISTINCT(event_type, 0, 30, days) CLASSIFY TOP 5` →
`multilabel_classification`. That is what `--row 6 --repair` runs, making rows 6
and 7 the controlled pair the matrix intended (§6).

## 3. The anchor edge is *not* a trap here, and that is itself the finding — NEW

NYC row 16 and BCCC row 14 established that `_find_keys` does
`keys = curr_keys` (**assign, not extend**), so the last-declared foreign key
wins silently, with no PQL syntax to override it.

`interactions` has two foreign keys, so the hazard looks like it applies. It does
not: `user_id → users` and `item_id → items` point at **different destination
tables**, so `_find_keys` has one candidate per destination. Compiling every query
under both declaration orders gives **byte-identical results**, and every run
asserts it:

```
interactions fkeys (declaration order): ['ITEM_ID', 'USER_ID']
resolved join key : interactions.USER_ID
task_type         : temporal_link_prediction (link_prediction)
rhs node type     : items
```

**The rule is not "two foreign keys are dangerous"; it is "two foreign keys into
the same table are dangerous."** BCCC and NYC both had the latter and read it as
the former. Yambda is the counter-example that separates them.

## 4. The no-model instruments, run before any training

### Saturation — rows 13/15 are the *least* saturated instance in the matrix

|                           | NYC row 16 |   BCCC row 14 |      **Yambda row 15** | **Yambda row 13** |
| ------------------------- | ---------: | ------------: | ---------------------: | ----------------: |
| candidate universe        |  264 zones | 407,244 hosts |    **1,958,664 items** |   1,958,664 items |
| median truth-set size     |    **215** |             1 | **64** (30-day window) |        168 (ever) |
| median truth / candidates |  **0.814** |     0.0000025 |          **0.0000327** |         0.0000858 |

Row 15's target is genuinely open, and `recall@100` has real headroom. This is a
well-posed ranking problem — the opposite of NYC row 16.

**Popularity is a weak baseline here.** The top 100 items cover **2.49%** of
user–item pairs; **964,557 of 1,958,664 items (49.2%) are touched by exactly one
user**. Compare BCCC, where the top 100 destinations covered 21.5%.

### Stability — the no-model ceiling, and it splits the block cleanly in two

How much of the next 30-day window's target set is already in the previous one:

| target                  | active users | mean target size | **mean recall** | mean Jaccard |     identical sets |
| ----------------------- | -----------: | ---------------: | --------------: | -----------: | -----------------: |
| `event_type` (rows 6/7) |       34,985 |             1.62 |      **0.7572** |       0.7785 | 16,569 (**47.4%**) |
| `item_id` (row 15)      |       34,985 |           108.98 |      **0.1613** |       0.0982 |      34 (**0.1%**) |
| *NOAA `element`*        |       29,601 |             3.39 |        *0.9402* |     *0.8729* |            *68.9%* |
| *NYC dropoff zone*      |          264 |              215 |        *0.9069* |            — |                  — |

**This is the sharpest contrast in the whole series.** Rows 6/7's target is 76%
already known from last month — the regime where a `GROUP BY` was unbeatable.
Row 15's is **84% new every month** — nothing in the matrix has looked like this
before, and it is why row 15 is the one row where the honest protocol produces
near-zero scores for *every* ranker (§7).

### Row 10's target, and its structural baselines over the whole pool

`played_ratio_pct` is bimodal and not a percentage:

|                                | measured over 194,185,872 labeled rows |
| ------------------------------ | -------------------------------------: |
| min / median / mean / sd / max |  0 / 100 / **64.56** / 44.10 / **159** |
| exactly 0                      |                      14,630,803 (7.5%) |
| exactly 100                    |                **102,825,011 (53.0%)** |

| baseline (whole pool)         |       MAE |          R² |
| ----------------------------- | --------: | ----------: |
| constant (labeled mean)       |     41.64 |      0.0000 |
| **`GROUP BY item_id` median** | **33.48** | **−0.3668** |
| **`GROUP BY user_id` median** | **27.32** | **−0.0450** |

**MAE and R² disagree in sign**, and that is the bimodality: a median is an
excellent L1 predictor of a two-spike distribution and a terrible L2 one. Any
report of row 10 quoting only one of the two is misleading. Correlation with
`track_length_seconds` is **−0.058** and with `is_organic` **−0.115**.

## 5. Row 10 — the model is a constant, and the null structure breaks the holdout

`y_tiny`, 20% holdout, 762,395 training rows, 2,115.4 s.

### The structural nulls are the story

Row 10's target is a **static column**, so its compiled prediction table carries
`{"compare": {"op": "is_null", ...}}`. But `played_ratio_pct` is *already* null
on **every non-`listen` row** (§1), so after blanking, `IS NULL` no longer means
"held out"; it means **"held out or structurally absent."**

```
predicted rows            : 222,509
  of which scoreable      : 190,943
  31,566 predicted rows have no truth -- the structurally-null non-listen events
```

**14.2% of the prediction set is unscoreable by construction.** A user who takes
`num_entities_predicted` at face value over-counts their evaluation set by 16%.

### The numbers

|                                 |       n |          MAE |     RMSE |     WAPE |         R² |
| ------------------------------- | ------: | -----------: | -------: | -------: | ---------: |
| **row 10 model**                | 190,943 |   **40.585** |   43.499 |   0.6176 | **0.0007** |
| *constant (labeled mean)*       | 190,943 |     *40.813* | *43.514* | *0.6211* |  *−0.0000* |
| ***`GROUP BY item_id` median*** | 190,943 | ***31.531*** | *48.393* | *0.4798* |  *−0.2368* |
| ***`GROUP BY user_id` median*** | 190,943 | ***28.015*** | *45.133* | *0.4263* |  *−0.0758* |

**The model is a constant predictor and this is not an interpretation.** The
prediction column has **sd 7.70** against a truth sd of 44.10; its median is
66.16 and its interquartile range is **66.15 → 66.17**.

Its R² of 0.0007 is nominally the *best* of the four — the only one above zero —
while its MAE is 45% worse than a one-line `GROUP BY user_id`. Both statements
are true and neither should be quoted alone.

### Why, mechanically

```
users            0 encoded columns: []
items            0 encoded columns: []
interactions     3 encoded columns: ['EVENT_TYPE', 'IS_ORGANIC', 'TRACK_LENGTH_SECONDS']
```

**Three features.** `EVENT_TIME` is a `timestamp` and therefore never a feature
(`kumo_core/pquery.py:810`). Of the three survivors, `EVENT_TYPE` is constant
(`listen`) on every scoreable row, and the other two correlate −0.058 and −0.115
with the target. The two neighbour tables contribute **zero** columns each, so a
user and an item are bare lookup embeddings. The `GROUP BY user_id` median that
beats it by 31% on MAE is exactly the one-hop aggregate the model was supposed to
learn.

## 6. Rows 6 and 7 — the `CLASSIFY`/`RANK` pair, and the difference is not measurable

Row 6 as published does not compile (§2), so the pair is run with the window row
7 already has — **byte-identical data, byte-identical training tables, one word
different.**

### The saturation caveat governs everything

|                                     |        `y_tiny` |       `y_smoke` |
| ----------------------------------- | --------------: | --------------: |
| **active in the target window**     | 180 (**84.9%**) | 909 (**84.2%**) |
| candidate classes                   |           **5** |           **5** |
| event types per user min/median/max |   1 / **1** / 5 |   1 / **1** / 5 |

The vocabulary is 5 and the cut-off is 5, so **`recall@5 = 1.0000` for
everything**. The comparison is purely an ordering comparison and only k = 1 and
2 can separate anything.

### `y_smoke` (909 scored users)

|                      |   k | prec@k |   recall@k |  **MAP@k** | NDCG@k |
| -------------------- | --: | -----: | ---------: | ---------: | -----: |
| **row 6 `CLASSIFY`** |   1 | 0.9923 |     0.7483 | **0.9923** | 0.9923 |
|                      |   5 | 0.3281 | **1.0000** |     0.9818 | 0.9904 |
| **row 7 `RANK`**     |   1 | 0.9923 |     0.7483 | **0.9923** | 0.9923 |
|                      |   5 | 0.3281 | **1.0000** |     0.9820 | 0.9905 |
| *popularity*         |   1 | 0.9318 |     0.7038 |   *0.9318* | 0.9318 |
| *persistence*        |   1 | 0.9888 |     0.7258 |   *0.9888* | 0.9888 |

**MAP@1 is 0.9923 for both.** The largest gap anywhere is 0.0007. The one metric
that moves is macro AUROC:

|                  | `y_tiny` | `y_tiny` repeat | `y_smoke` |
| ---------------- | -------: | --------------: | --------: |
| row 6 `CLASSIFY` |   0.7599 |               — |    0.7292 |
| row 7 `RANK`     |   0.7417 |      **0.7220** |    0.7527 |

The row-7 self-noise band is **0.7417 → 0.7220, a swing of 0.0197**. Every
`CLASSIFY`-vs-`RANK` gap is smaller than that and the signs disagree between
tiers. **Verdict: no measurable quality difference** — reproducing NOAA's
conclusion on a second fixture, with five classes instead of sixty-one.

Both clear popularity at k = 1 by ~0.06 MAP and both **tie persistence** (0.9888
against 0.9923). Given §4's 0.7572 stability figure that is exactly what should
have been expected.

### What *does* differ

|                              | `CLASSIFY` | `RANK`              |
| ---------------------------- | ---------- | ------------------- |
| reported metric names        | **22**     | **13**              |
| overlap between the two sets | **zero**   |                     |
| `PREDICTED` column           | present    | **absent**          |
| train, `y_smoke`             | 326.7 s    | **480.2 s (1.47×)** |

NOAA measured `RANK` at 2.6× `CLASSIFY`'s wall clock; here it is 1.10× at
`y_tiny` and 1.47× at `y_smoke` — real but smaller, and growing with the tier.

**A new symptom of the hardcoded `k` set.** Row 7's reported metrics on a
five-class problem include `test_precision@10 = 0.1691`, `test_recall@10 = 1.0000` and `test_map@100 = test_map@10`. There is no tenth or hundredth class.
NYC found the k-set hardcoded from the "too coarse" side; this is the "too fine"
side.

## 7. Rows 13 and 15 — the static/temporal LP pair, and both lose to a `GROUP BY`

Two protocols, as in BCCC row 14: **unfiltered** and **filtered** (everything the
user already reaches is struck first — the standard LP protocol, and **the honest
column**). Three no-model baselines: **popularity**, **co-listen** (item–item
co-occurrence — the `GROUP BY` that uses exactly the structure the GNN is being
paid to learn) and, for row 15, **persistence**.

### Row 15 (temporal) — the model's entire score is re-listening

`y_smoke`: 296,981 candidates, 909 users, median 66 truth items.

|                   |   k | prec@k |   recall@k |    **MAP@k** |     NDCG@k |
| ----------------- | --: | -----: | ---------: | -----------: | ---------: |
| **unfiltered**    |     |        |            |              |            |
| row 15 model      |   1 | 0.2717 |     0.0083 |   **0.2717** |     0.2717 |
|                   | 100 | 0.0589 |     0.0938 |   **0.0560** |     0.1276 |
| ***persistence*** |   1 | 0.2706 |     0.0060 | ***0.2706*** |     0.2706 |
|                   | 100 | 0.1568 | **0.2030** | ***0.1130*** | **0.2284** |
| *co-listen*       | 100 | 0.1106 |     0.1362 |     *0.0478* |     0.1427 |
| *popularity*      | 100 | 0.0288 |     0.0334 |     *0.0086* |     0.0415 |
| **filtered**      |     |        |            |              |            |
| **row 15 model**  |   1 | 0.0110 |     0.0003 |   **0.0110** |     0.0110 |
|                   | 100 | 0.0079 |     0.0054 |   **0.0012** |     0.0099 |
| *popularity*      | 100 | 0.0069 |     0.0092 |     *0.0017* |     0.0104 |
| *co-listen*       | 100 | 0.0083 |     0.0096 |     *0.0020* |     0.0117 |

**Unfiltered MAP@1 0.2717 → filtered 0.0110.** 96% of the model's apparent
performance is proposing tracks the user already plays. Persistence — literally
"rank the user's own history by play count" — matches it at k = 1 and beats it
**2.0× at MAP@100**.

Under the honest protocol **everything collapses**, and the model sits *below*
both trivial baselines. Filtered `recall@100` is 0.0054 — 100 guesses out of
296,981 candidates find half a percent of the ~85 genuinely new tracks a user
picks up each month.

This is **not** NYC row 16's failure mode. NYC's task was degenerate; this one is
well-posed (§4) and simply hard. Given that 84% of next month's tracks are new to
the user, a model with three interaction features and two featureless node tables
has nothing to work with.

### Row 13 (static) — beaten by a co-occurrence `GROUP BY` by 2.4×, at both tiers

`y_smoke`: 296,981 candidates, 1,079 users, median 193 held-out items.

|                  |   k |     prec@k |   recall@k |    **MAP@k** |     NDCG@k |
| ---------------- | --: | ---------: | ---------: | -----------: | ---------: |
| **row 13 model** |   1 |     0.2505 |     0.0021 |   **0.2505** |     0.2505 |
|                  | 100 |     0.1233 |     0.0428 |   **0.0275** |     0.1487 |
| *popularity*     |   1 |     0.3284 |     0.0027 |     *0.3284* |     0.3284 |
|                  | 100 |     0.1036 |     0.0581 |     *0.0428* |     0.1347 |
| ***co-listen***  |   1 | **0.4082** |     0.0039 | ***0.4082*** | **0.4082** |
|                  | 100 |     0.1433 | **0.0868** | ***0.0673*** | **0.1858** |

**The model loses to co-listen at every cut-off — 2.4× at MAP@100 — and it also
loses to plain global popularity.**

| tier      | scored users |  model | *popularity* | ***co-listen*** |
| --------- | -----------: | -----: | -----------: | --------------: |
| `y_tiny`  |          212 | 0.0205 |     *0.0358* |    ***0.0484*** |
| `y_smoke` |        1,079 | 0.0275 |     *0.0428* |    ***0.0673*** |

**The ratio is stable at 2.4× across a 5× scale-up.** This is BCCC row 14's
result — model 2.9× below a co-source filter — reproduced on a completely
different domain, graph shape and saturation level. **Two independent fixtures
now say the same thing about static link prediction on this backend.**

For row 13 the *filtered* numbers are **higher** than the unfiltered ones, the
opposite of row 15 — the holdout doing its job, since the held-out pairs are
precisely the items removed from the graph.

### Why, mechanically — the same explanation on both rows

```
users            0 encoded columns: []
items            0 encoded columns: []
interactions     4 encoded columns: [EVENT_TYPE, IS_ORGANIC, PLAYED_RATIO_PCT, TRACK_LENGTH_SECONDS]
```

`items` is **both** the RHS candidate table and featureless, exactly as `hosts`
was for BCCC row 14 and `zones` for NYC row 16. Co-listen is the 2-hop count the
GNN is failing to aggregate, and it is one self-join.

### The split, measured rather than assumed

Both rows get a **link** split, not an entity split:

```
row 13  train rows=212 target links=197,437 excluded keys=0
        val   rows=212 target links=24,856  excluded keys=197,437
        test  rows=212 target links=24,488  excluded keys=222,293
        entities in both train and test: 212 of 212 -- a LINK split
```

But **the val and test edges are still in the DiskGraph index** — the index is
built once and the split happens afterwards — so the standard transductive leak
is present, which is the other reason both rows are scored against their own
holdout here.

## 8. The noise floor is 6% on this fixture, not BCCC's 38% — NEW

Row 15 at `y_tiny`, two runs, identical tier / query / plan:

|                         |  run A |  run B | difference |
| ----------------------- | -----: | -----: | ---------: |
| reported `test_map@100` | 0.0607 | 0.0603 |  **−0.7%** |
| unfiltered MAP@100      | 0.0538 | 0.0569 |  **+5.8%** |
| filtered MAP@100        | 0.0005 | 0.0006 |       +20% |

Row 7 at `y_tiny`: MAP@1 **0.9944 both times**.

**The unfiltered LP estimate is stable to about 6%**, an order of magnitude
better than BCCC's. The mechanism is visible in the split: row 15's val stage
carries **18,260 target links over 180 entities** where BCCC's whole test stage
carried ~1,300 over 404. More links per entity is a lower-variance estimate.

So §7's margins — **2.0× for persistence on row 15 and 2.4× for co-listen on row
13** — are well outside the noise band and are real. The *filtered* row 15
numbers are **not**: all three are inside 20% of each other on a metric whose own
repeat swing is 20%, so **at the filtered protocol the model, popularity and
co-listen are indistinguishable, and all three are near zero.**

## 9. Cost — the first fixture where disk would also fail

|                                              |                             measured |
| -------------------------------------------- | -----------------------------------: |
| stage → Snowflake, 4 of 96 parts (200M rows) | **440.6 s** on `KUMO_L` (2.2 µs/row) |
| **DiskGraph index, with a time column**      |       **50.9 bytes per interaction** |
| **DiskGraph index, no time column**          |       **39.0 bytes per interaction** |
| generated training table                     |                  **1.03 KB per row** |
| peak RSS, LP rows                            |                2.05 GB → **2.75 GB** |
| peak `/mnt/localssd` across all runs         |         **7.12 GB** of 217 available |

The index size is consistent to within 1% across a 5× scale change. Declaring the
time column costs **~12 bytes per row** — a measurable price on the row 13 / row
15 choice.

**Disk would not hold the full fixture.** 4,785,859,887 × 50.9 B = **243.6 GB**
against **217 GB** free. Every previous block concluded "disk is not the
constraint"; this is the first where the index alone does not fit, by 12%.

**Memory binds far earlier.** Marginal RSS is **~0.17 GB per million
interactions** on the LP rows, so 15 GB extrapolates to roughly **75–90 million
interactions — about 1.8% of the fixture**, or 16,000–19,000 users against
1,000,000. **No tier here is the published problem, and no tier reachable on this
box would be.** Linear extrapolation from two points.

Time would not have bound first: row 15 went 166.0 s → 453.6 s for 5.2× the data,
an exponent of **0.63 — sublinear**, because a bigger Yambda tier buys users, not
anchors. That is the inverse of NOAA, where a bigger tier widened the anchor span
and drove the exponent to 1.18. **Yambda is memory-bound and nothing else.**

Row 10 is the exception: **2,115.4 s** at `y_tiny`, because its entity is the
fact table, so its training table is 762,395 rows where the others' are 10,600.
At 1.03 KB/row its training table alone would be **4.9 TB** at full scale.

## 10. Tiering and the two holdouts

**A tier is a hash sample of users, not of interactions.** Four of five rows have
`users.user_id` as the entity, so thinning a user's interactions changes that
user's label. Row 10's entity *is* the fact table, where a row sample would have
been correct; it is run on the same user tiers anyway, because a uniform row
sample would leave every user with 1/n of their history as graph context.
**Stated so the choice is visible rather than inherited.**

| tier        | bucket |       users |    interactions | distinct items |
| ----------- | -----: | ----------: | --------------: | -------------: |
| `y_tiny`    |     50 |         212 |         984,904 |        117,378 |
| `y_smoke`   |    250 |       1,079 |       5,097,595 |        296,981 |
| `y_pool`    | 10,000 |      41,975 |     199,962,908 |      1,958,664 |
| *(fixture)* |      — | *1,000,000* | *4,785,859,887* |    *9,390,623* |

**Yambda's entities are two orders of magnitude denser than any previous
fixture's** — 4,786 interactions per user. That is why the user counts look so
small: 212 users is a million interactions.

`items` is narrowed per tier to the items the tier's interactions actually reach.
Carrying all 9,390,623 published items would make every candidate outside the tier
an isolated, featureless node — BCCC §6's unreachable-truth failure from the other
direction.

**Row 10's holdout** blanks on a *salted* hash of `interaction_id`, because NYC
nested two thresholds on one hash and `MOD(h, 10000) < 5` implies
`MOD(h, 100) < 5`.

**Row 13's holdout** is over distinct `(user, item)` pairs, with both of BCCC
§6's invariants — including **every held-out item keeps at least one other
user**. The build asserts it:

```
held-out items unreachable in the graph: 0  (0.00% of truth)  OK
```

## 11. Carried over, confirmed again on five more task types

- **`training_job.metrics` is still ignored**, and both directions fired in one
  block. Row 10 asked for five metrics and got **four** — `r2` and `wape` never
  computed. Row 6 asked for two and got **22**; row 7 asked for three and got
  **13**. `kumo_core/training/workflow.py::_resolved_task_options` reads the list
  from a `task` model-plan section `ModelPlan` does not have. **A check that only
  looks for missing names calls rows 6 and 7 "applied" and only row 10 catches
  it.**
- **`tune_metric` does arrive** on all five task types; the fix in the working
  tree holds.
- **`RANK TOP k` has no effect on the prediction output.** Rows 13/15 asked for
  `TOP 100` and returned exactly 100 — which looks like compliance and is
  coincidence. Rows 6/7 asked for `TOP 5` and returned 5, also coincidence: the
  vocabulary is 5.
- **No `timestamp` column is ever a feature** (`pquery.py:810`). On this fixture
  that costs more than usual — hour-of-day and day-of-week are the obvious
  features for music listening and none exist.
- **`num_entities_predicted` reports rows, not entities**: 21,200 for 212 users.
- **`validation` metrics come back empty** — a seventh fixture, on all five task
  types.
- **`scikit-learn` absent** → `Optuna search analysis failed` on every run.
- **`_column_stats` stat-token collisions: none.** All eight column names are
  clean, checked rather than inherited.
- **`Table.infer_metadata` discards stypes set before it.** `USER_ID`/`ITEM_ID`
  are load-bearing — `stype_inference` calls an integer column *categorical*
  under 4,000 distinct values, which a small tier's user column satisfies easily,
  and a categorical foreign key is not a foreign key: rows 13/15 would silently
  stop being link prediction.
- **`SnowflakeConnector(warehouse=...)` is inert locally**; `--no-publish-snapshots`
  on every run; **the in-tree SDK on `PYTHONPATH` is not optional.**

### New to this block

- **A fixture's published null structure can silently redefine a static-column
  holdout** — 14.2% of row 10's prediction set has no truth and never could (§5).
- **All four cells of `time_range_validator`'s 2×2 on one fixture** (§2).
- **The `_find_keys` hazard needs two foreign keys into the *same* table**, not
  merely two foreign keys (§3).
- **The reported `@10`/`@100` metrics are meaningless on a small vocabulary**
  (§6).
- **Declaring a time column costs ~12 bytes per row of index** (§9).
- **The LP noise floor is fixture-dependent**: 6% here against BCCC's 38%,
  explained by links-per-entity in the val stage (§8).

## Reproducing

```bash
source scripts/_env.sh  # resolves the monorepo SDK and root .venv

# STEP ZERO: does the fixture exist?
"$KUMO_SDK_PYTHON" scripts/kaggle/yambda/probe_yambda_fixture.py

# What all five rows compile to, under both fkey orders and with/without the
# time column. One second, no data -- establishes §2 and §3.
"$KUMO_SDK_PYTHON" scripts/kaggle/yambda/probe_yambda_query.py \
    --fkey-order both --no-time-column --variants

# event_time's unit, the vocabulary, the null structure, the part layout.
"$KUMO_SDK_PYTHON" scripts/kaggle/yambda/probe_yambda_raw.py

# Read four of the 96 parts into Snowflake. ~440 s, once.
"$KUMO_SDK_PYTHON" scripts/kaggle/yambda/build_yambda_tables.py --raw-only

# Every no-model instrument -- this is what decides the block (§4).
"$KUMO_SDK_PYTHON" scripts/kaggle/yambda/probe_yambda_stats.py --all

# Build one tier and run one row. Scales: y_tiny | y_smoke | y_small | y_part | y_pool.
scripts/kaggle/yambda/run_yambda_tier.sh 10 y_tiny     # regression
scripts/kaggle/yambda/run_yambda_tier.sh 7  y_tiny     # multilabel ranking
scripts/kaggle/yambda/run_yambda_tier.sh 15 y_tiny     # temporal LP
scripts/kaggle/yambda/run_yambda_tier.sh 13 y_tiny     # static LP
TAG=_repair scripts/kaggle/yambda/run_yambda_tier.sh 6 y_tiny --repair

# The two deliberate rejections (§2), both before ingest.
"$KUMO_SDK_PYTHON" scripts/kaggle/yambda/run_yambda_local.py \
    --row 6 --scale y_tiny --no-publish-snapshots
"$KUMO_SDK_PYTHON" scripts/kaggle/yambda/run_yambda_local.py \
    --row 13 --scale y_tiny --time-column --no-publish-snapshots

# The noise floor (§8): a byte-identical repeat.
TAG=_b scripts/kaggle/yambda/run_yambda_tier.sh 15 y_tiny
```

Recovery, if a run is interrupted or OOM-killed during ingest (no
`_phase_complete_context` marker is ever written, so every resume fails):

```bash
rm -rf /mnt/localssd/kumo-yambda/scratch/diskgraph_snapshots/<graph key>
```

Snowflake objects (account `TLB22710`, schema `HYU.YAMBDA`): `INTERACTIONS_RAW`
(199,962,908 rows), `USERS_RAW`, `ITEMS_RAW`, plus per-tier `INTERACTIONS_<TIER>`,
`USERS_<TIER>`, `ITEMS_<TIER>`, `INTERACTIONS_<TIER>_H20` + `_TRUTH`, and
`INTERACTIONS_<TIER>_LP` + `_TRUTH` + `ITEMS_<TIER>_LP`.

## Scripts

| file                      | purpose                                                                                                                 |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `probe_yambda_fixture.py` | **step zero** — does the fixture exist, and what does the manifest say (§1)                                             |
| `probe_yambda_query.py`   | what all five rows compile to, before any data exists (§2, §3)                                                          |
| `probe_yambda_raw.py`     | `event_time`'s unit, the vocabulary, the null structure, the part layout (§1)                                           |
| `probe_yambda_stats.py`   | saturation, stability, popularity concentration, row 10's structural baselines (§4)                                     |
| `build_yambda_tables.py`  | the four-part raw read, the user tiers, and the two holdouts (§10)                                                      |
| `run_yambda_local.py`     | the end-to-end run for all five rows, with resolution assertions and filtered/unfiltered scoring against every baseline |
| `run_yambda_tier.sh`      | build + run one row on one tier unattended, sampling free disk and RSS                                                  |
