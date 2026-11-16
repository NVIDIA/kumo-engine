# NOAA GHCN-Daily on the SDK's local backend — matrix rows 5, 8 and 11

One fixture, three matrix rows, run in-process the same way
[row 1](../ethereum/FINDINGS.md), [row 2](../iot23/FINDINGS.md),
[row 3](../bccc/FINDINGS.md) and [row 4](../iot23_detailed/FINDINGS.md) were:
Snowflake catalog reads, DiskGraph ingest, `kumo_core` AutoML, batch scoring,
no Kumo service.

| matrix row | task                      | PQL as published                                                                                                                        |
| ---------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 5          | multilabel classification | `PREDICT LIST_DISTINCT(observations.element)` / `CLASSIFY TOP 25` / `FOR EACH stations.station_id`                                      |
| 8          | multilabel ranking        | the same with `RANK TOP 25`                                                                                                             |
| 11         | forecasting               | `PREDICT AVG(observations.value WHERE observations.element = 'TMAX', 0, 7, days)` / `FOR EACH stations.station_id`, `forecast_length=4` |

**None of the three queries runs as published.** Rows 5 and 8 are rejected by
the PQL validator (§2); row 11 parses but silently compiles to `regression`
rather than `forecasting` (§3). All three are established before any training.

With minimal repairs — a time window for rows 5 and 8, a `FORECAST 4 TIMEFRAMES`
clause for row 11 — the verdict per row:

| row                             | verdict        | scale              | headline                                                                                                                                 |
| ------------------------------- | -------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **5** multilabel classification | **end to end** | `tiny` and `smoke` | MAP@5 0.9832 at `smoke` against a **persistence baseline of 0.9761** (§4, §8)                                                            |
| **8** multilabel ranking        | **end to end** | `tiny`             | statistically identical to row 5 on the same data, at **2.6× the training time** (§8)                                                    |
| **11** forecasting              | **blocked**    | —                  | two backend defects: horizon > 1 empties the val/test split, and batch prediction is unimplemented for forecasting at *any* horizon (§9) |

Three defects are new and none is fixture-specific: the model-plan drop whose
root cause rows 3 and 4 could not isolate is found and named — and was fixed in
the working tree by another session partway through this block, which §7 records
per run — forecasting cannot complete at all (§9), and an interrupted ingest
wedges a graph permanently (§10).

## Runs

All three rows were run on the `tiny` tier (231 stations, 4,862,394
observations, 61 elements); row 5 was also run on `smoke` (1,150 stations,
27,158,157 observations, 81 elements). Rows 5 and 8 read **byte-identical
training tables** at `tiny` — 419,958 rows, 1,818 anchors, 61 target classes,
confirmed from `stats.json` on both runs — so the difference between them is
`CLASSIFY` versus `RANK` and nothing else.

|                                    |                    **row 5** `CLASSIFY` tiny |             **row 8** `RANK` tiny | **row 11** forecasting tiny |  **row 5** `CLASSIFY` smoke |
| ---------------------------------- | -------------------------------------------: | --------------------------------: | --------------------------: | --------------------------: |
| task type                          |                  `multilabel_classification` |              `multilabel_ranking` |               `forecasting` | `multilabel_classification` |
| end to end                         |                                      **yes** |                           **yes** |                 **no — §9** |                     **yes** |
| stations / observations            |                                  231 / 4.86M |                       231 / 4.86M |                 231 / 4.86M |              1,150 / 27.16M |
| AutoML trials                      |                                            4 |                             **8** |               8, all failed |                           4 |
| train                              |            **1,753.3 s** (2,904.5 s re-run†) |                     **4,543.6 s** |             285.1 s to fail |              **12,129.1 s** |
| predict                            |                                        0.7 s |                             0.7 s |                           — |                       1.8 s |
| stype-inference scan (`connector`) |                                        3.5 s |                             3.5 s |                       5.2 s |                 **131.3 s** |
| DiskGraph ingest                   |                   **4.78 s**, 109 MB written |                            reused |                      reused | **23.97 s**, 714 MB written |
| training-table rows                |                                      419,958 |                           419,958 |                     114,812 |               **2,158,550** |
| anchors × entities                 |                                  1,818 × 231 |                       1,818 × 231 |                6,819 weekly |               1,877 × 1,150 |
| classes                            |                                           61 |                                61 |                           — |                          81 |
| prediction rows                    |                            14,091 (231 × 61) |                 14,091 (231 × 61) |                           — |         93,150 (1,150 × 81) |
| prediction columns                 | `ENTITY, TIMESTAMP, PREDICTED, CLASS, SCORE` | `ENTITY, TIMESTAMP, CLASS, SCORE` |                           — |          same as row 5 tiny |
| peak RSS                           |                                      ~2.1 GB |                           2.11 GB |                     2.11 GB |           **7.74 GB** of 15 |
| reported `test_ap_macro`           |                             0.8457 (0.8693†) |                                 — |                           — |                  **0.9283** |
| reported `test_auroc_macro`        |                             0.9905 (0.9931†) |                                 — |                           — |                  **0.9966** |
| model plan applied                 |                                  **no** (§7) |                       **no** (§7) |                         yes |                **yes** (§7) |

† A re-run of row 5 at `tiny` after the model-plan boundary was fixed in the
working tree partway through this block (§7). Same query, same tier, same data;
only the tune metric differs (`auroc_macro` instead of the default `ap_macro`).

Scored independently against ground truth recomputed in SQL for the 30 days
after the anchor 2019-05-04. Roughly three quarters of the entity set is dead by
2019 and has an **empty** target, so it is excluded from every rate (§4).

**tiny** — 57 of 231 stations scored, 174 (75.3%) empty; 21 labels with both
classes present. `row 5†` is the same query and tier re-run after the model-plan
fix landed mid-sequence, so it tunes on `auroc_macro` instead of the default
`ap_macro` (§7):

|                  |   k | prec@k | recall@k |  **MAP@k** | **NDCG@k** |
| ---------------- | --: | -----: | -------: | ---------: | ---------: |
| **row 5**        |   1 | 1.0000 |   0.4228 | **1.0000** | **1.0000** |
|                  |   3 | 0.8012 |   0.8196 | **0.9795** | **0.9850** |
|                  |   5 | 0.5965 |   0.9287 | **0.9843** | **0.9913** |
|                  |  25 | 0.1474 |   1.0000 | **0.9879** | **0.9951** |
| **row 5†**       |   1 | 1.0000 |   0.4228 | **1.0000** | **1.0000** |
| *(plan applied)* |   3 | 0.8012 |   0.8196 | **0.9795** | **0.9850** |
|                  |   5 | 0.5965 |   0.9287 | **0.9834** | **0.9908** |
|                  |  25 | 0.1474 |   1.0000 | **0.9869** | **0.9946** |
| **row 8**        |   1 | 1.0000 |   0.4228 | **1.0000** | **1.0000** |
|                  |   3 | 0.7953 |   0.8167 | **0.9737** | **0.9809** |
|                  |   5 | 0.6000 |   0.9313 | **0.9846** | **0.9921** |
|                  |  25 | 0.1474 |   1.0000 | **0.9859** | **0.9940** |
| *persistence*    |   1 | 1.0000 |   0.4228 |   *1.0000* |   *1.0000* |
|                  |   3 | 0.8567 |   0.8211 |   *0.9795* |   *0.9850* |
|                  |   5 | 0.7921 |   0.9270 |   *0.9873* |   *0.9917* |
|                  |  25 | 0.7531 |   0.9898 |   *0.9858* |   *0.9907* |
| *popularity*     |   1 | 0.9123 |   0.3819 |   *0.9123* |   *0.9123* |
|                  |   5 | 0.4702 |   0.7903 |   *0.7380* |   *0.8199* |
|                  |  25 | 0.1754 |   1.0000 |   *0.8047* |   *0.8989* |

macro AUROC: **row 5 0.9806**, **row 5† 0.9827**, **row 8 0.9746**.

**smoke** — 299 of 1,150 stations scored, 851 (74.0%) empty; 28 labels with both
classes present:

|               |   k | prec@k | recall@k |  **MAP@k** | **NDCG@k** |
| ------------- | --: | -----: | -------: | ---------: | ---------: |
| **row 5**     |   1 | 0.9933 |   0.4593 | **0.9933** | **0.9933** |
|               |   3 | 0.7625 |   0.8484 | **0.9729** | **0.9800** |
|               |   5 | 0.5625 |   0.9525 | **0.9832** | **0.9903** |
|               |  25 | 0.1296 |   1.0000 | **0.9856** | **0.9926** |
| *persistence* |   1 | 0.9967 |   0.4604 |   *0.9967* |   *0.9967* |
|               |   3 | 0.8372 |   0.8406 |   *0.9667* |   *0.9792* |
|               |   5 | 0.7538 |   0.9414 |   *0.9761* |   *0.9886* |
|               |  25 | 0.7081 |   0.9886 |   *0.9781* |   *0.9906* |
| *popularity*  |   1 | 0.9264 |   0.4298 |   *0.9264* |   *0.9264* |
|               |   5 | 0.4482 |   0.8190 |   *0.7726* |   *0.8420* |
|               |  25 | 0.1292 |   0.9982 |   *0.8322* |   *0.9126* |

macro AUROC: **row 5 0.9884**.

**Read the persistence rows, not the model rows.** At `tiny` no model clears
persistence by anything that survives 57 stations of noise. At `smoke` row 5
does clear it, at every k above 1, by 0.006–0.008 MAP — small, but consistent in
sign across four cut-offs and two metrics, on 299 stations. §8 is the
comparison; §4 is why the baseline is so high.

The `row 5†` re-run exists to check that the `tiny` → `smoke` difference is
scale and not the tune metric, because the model-plan fix landed between the two
(§7). It is not: applying the plan moves MAP@5 by **−0.0009**, MAP@25 by
**−0.0010** and macro AUROC by **+0.0021** on identical data — noise on 57
stations. **The dropped model plan cost this fixture nothing measurable**, which
is the same conclusion rows 3 and 4 reached from the other direction, now
measured rather than argued.

Every run was uncontended — nothing else occupied the 4 vCPUs — so the timings
are measurements rather than upper bounds.

Disk is not a constraint anywhere near these scales, and would not be at full
scale either. The index DiskGraph writes is **22.5 bytes per observation** at
`tiny` (109.2 MB for 4.86M) and **26.3** at `smoke` (714.4 MB for 27.16M), so
all 2,895,739,748 observations would index into roughly **65–76 GB** against
216 GB free. The whole run footprint on `/mnt/localssd` never exceeded
**1.4 GiB** across both tiers and every row, against 216 GiB available — three
orders of magnitude of headroom, and the first fixture in the series where disk
was never worth watching. §6 is why that does not help.

**Largest tier reached: `smoke`, for row 5.** Scaling `tiny` → `smoke` is
**5.14× the training-table rows for 6.92× the time** (419,958 → 2,158,550 rows,
1,753.3 s → 12,129.1 s), an exponent of **1.18** — *superlinear*, because the
tier also widens the anchor span (§6). Carrying 1.18 forward to `full`'s
~355 million training rows gives on the order of **10⁴ hours**; the honest
statement is simply that full scale is unreachable, and §6 says why it is AutoML
rather than ingest or disk that makes it so.

## 1. The fixture matches the matrix, which is a first

Four fixtures in, this is the only one with nothing to correct.

| claim                                                | measured                                                                                                                     |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 2,895,853,690 rows                                   | **2,895,739,748** `observations` + **113,942** `stations` = 2,895,853,690 — the matrix figure is the **sum**, as in rows 1–4 |
| 12.73 GB stored                                      | 12,733,092,223 B over 584 files (`LIST @stage`), against the manifest's 12,733,091,166                                       |
| key `stations.station_id`                            | **unique**: 113,942 rows, 113,942 distinct, 0 null. No dedup needed (contrast rows 1 and 3)                                  |
| edge `observations.station_id → stations.station_id` | **clean both ways**: 113,942 distinct ids referenced, 0 orphans, **0 stations with no observations**                         |
| `observation_date` as event time                     | microseconds since epoch, **signed**                                                                                         |
| tenths of degrees for TMAX                           | confirmed — §4                                                                                                               |

One pass over the 580 staged part files into a typed Snowflake table cost
**250.8 s on `KUMO_L`** — far cheaper than row 2's 600–1300 s for a tenth of the
rows, because NOAA is eight narrow columns and iot23 is twenty-one. Everything
after it is a scan of a native table.

Two structural facts drive the rest.

- **`stations` is one column wide.** Nothing but the key, exactly like `hosts`
  in rows 2–4 — but here it is also the *entity* table. There are no station
  features at all: no latitude, no elevation, no country. Every prediction comes
  from the observation history through the single edge.
- **`(station_id, event_time, element)` is unique** — 0 duplicate triples in
  2.9 billion rows — so the multilabel target is well defined.

### The span is 256 years and the density is not remotely uniform

`event_time` runs **1763-01-01 → 2019-06-03**, 0 null. It is the first fixture
in the series whose timestamps are *negative* over most of their span, which is
why `describe_split` takes `abs()` before inferring the unit.

| decade | observations | active stations |
| ------ | -----------: | --------------: |
| 1760s  |        5,114 |               1 |
| 1860s  |      356,656 |             159 |
| 1900s  |   64,693,710 |          12,987 |
| 1950s  |  264,360,397 |          37,339 |
| 1970s  |  335,285,317 |          44,289 |
| 2000s  |  369,177,203 |          50,208 |
| 2010s  |  335,230,911 |          61,384 |

Half the rows are after 1978, 20% after 2003, 10% after 2011. And the station
population is not stable: **52,558 of 113,942 stations last reported before
2010**, so at a 2019 anchor roughly three quarters of the entity set is dead.
§6 is what that does to the training table.

### Part files are time-ordered, which is a third pattern

Row 1's parts each spanned the whole history (a part was a 10% sample); row 2's
were one capture each. NOAA's are neither — they are **globally sorted by date**:

| part         |      rows | stations | station-id range         | date range              |
| ------------ | --------: | -------: | ------------------------ | ----------------------- |
| `part-00000` | 5,000,000 |    1,698 | AGE00135039..UZM00038618 | 1763-01-01 → 1884-12-15 |
| `part-00100` | 5,000,000 |   21,642 | AE000041196..WZ004834260 | 1945-04-25 → 1945-08-15 |
| `part-00289` | 5,000,000 |   33,383 | AE000041196..ZI000067991 | 1978-02-18 → 1978-04-14 |
| `part-00450` | 5,000,000 |   28,401 | AE000041196..ZI000067983 | 2001-08-31 → 2001-10-21 |
| `part-00579` |   739,748 |   26,415 | AE000041196..ZI000067975 | 2019-05-23 → 2019-06-03 |

A prefix slice is therefore the eighteenth century. Tiers are hash-sampled, and
on `station_id` rather than on a row key — §5.

### Columns

| column             |  non-null | distinct | kept?                                 |
| ------------------ | --------: | -------: | ------------------------------------- |
| `station_id`       |      100% |  113,942 | key / edge                            |
| `observation_date` |      100% |        — | event time                            |
| `element`          |      100% |  **137** | the row 5/8 target                    |
| `value`            |      100% |   11,979 | the row 11 target                     |
| `source_flag`      |      100% |       30 | dropped — provenance, not measurement |
| `measurement_flag` |    10.12% |        7 | dropped                               |
| `observation_time` |    20.80% |       70 | dropped                               |
| `quality_flag`     | **0.31%** |       14 | dropped                               |

`quality_flag` is the row 4 `SERVICE` shape exactly — non-null in 0.31% of rows,
so a 10,000-row inference sample can come back entirely null, which yields no
statistics, which is a hard error rather than a default. All four are
materialized in the tier tables and dropped in the run (`--with-flags` puts
them back), so nothing has to re-read the stage to revisit the decision. That is
row 4 §2's lesson applied: a narrow re-read of a variant-typed stage costs as
much as a wide one, so the first pass takes everything.

**No stat-token collisions.** Row 3 §6's `_column_stats` defect
(`stat_type.value in key`, matched against the whole key with the column name in
it) is confirmed still present in the installed `kumo_core` and confirmed inert
here: none of the eight column names contains a `ColStatType` token. Checked on
every run rather than inherited.

## 2. Rows 5 and 8 as published are rejected by the validator — NEW

`PREDICT LIST_DISTINCT(observations.element) CLASSIFY TOP 25 FOR EACH stations.station_id` does not validate. Neither does the `RANK` variant. Both
fail before any data is touched, and they fail for *different* reasons:

```
row 5  CLASSIFY  Aggregation missing a time range. TARGET aggregation
                 LIST_DISTINCT(observations.ELEMENT) is missing a time range.
                 At the moment, static usage of LIST_DISTINCT is only possible
                 for RANK problems.
row 8  RANK      ... static usage of LIST_DISTINCT is only possible for the
                 target clause with a foreign key.
```

The rule is written out as a four-case matrix in
`kumopql/validator/time_range_validator.py`, and one case is legal:

|              | foreign-key target                   | non-foreign-key target |
| ------------ | ------------------------------------ | ---------------------- |
| **RANK**     | **allowed** (static link prediction) | rejected               |
| **CLASSIFY** | rejected                             | rejected               |

`observations.element` is a categorical column, not a foreign key, so rows 5 and
8 land in the two rejected cells. A second, independent check in
`kumopql/validator/predictive_query_validator.py` rejects `LIST_DISTINCT` +
`CLASSIFY` with no time range outright, and two more reject any static
`LIST_DISTINCT` while a time column exists anywhere in the graph — *"likely to
result in an incorrect data split and information leakage"*, which is a fair
description of what the published query would have asked for.

**There is no legal static aggregate target on this graph at all.** Measured,
not inferred: `probe_noaa_query.py --variants` recompiles against a copy of the
graph with the fact table's time column removed, which is the only objection
that could be removed by editing the graph. Both survivors are still rejected —

```
AVG(observations.VALUE WHERE ...)            static usage aggregations is only
                                             supported for LIST_DISTINCT + RANK
LIST_DISTINCT(observations.STATION_ID) RANK  Target creates a loop by pointing
                                             to the entity table
```

— because `observations` has exactly one foreign key and it points at
`stations`, which is the entity. On a two-table dimension/fact graph the one
legal static form is unreachable by construction.

**This is the negative result the static-target holdout question wanted.**
Rows 1–4 all needed `--holdout-pct`, because a static *column* target compiles
its prediction table to `WHERE target IS NULL` and a fully-labeled fixture then
selects nothing (row 1 §1). Rows 5, 8 and 11 need **no holdout at all**, and
for a stronger reason than "the target is an aggregate": as *temporal* queries
they get no prediction-mode filter whatsoever — `diskgraph/compiler.py` applies
the entity filter only when `is_static_query`, so every entity is emitted.

**Measured, not inferred.** Batch prediction on the `tiny` tier returned scores
for **231 of 231 stations** and on `smoke` for **1,150 of 1,150** — the entire
entity table, including the three quarters of it that had stopped reporting
decades earlier. Nothing was blanked and nothing had to be. The holdout problem
of rows 1–4 is therefore confirmed **specific to static column targets**, and it
has no counterpart here.

For completeness, the static-aggregate path in the same compiler does have a
filter, and it is not `IS NULL`: it keeps entities whose `has_edges` into the
target table is `False`, i.e. only entities with *no* related rows. Had rows 5
and 8 been expressible as published, they would have failed the same way rows
1–4 did — harder, since no holdout could fix it. That branch is read off
`diskgraph/compiler.py`; it is **not** reachable on this graph, so it is
reported as a code reading rather than a measurement.

### The repair, and what it costs

Both rows are run with a time window added and nothing else changed:

```
PREDICT LIST_DISTINCT(observations.ELEMENT, 0, 30, days)
CLASSIFY TOP 25            (row 5)   /   RANK TOP 25   (row 8)
FOR EACH stations.STATION_ID
```

`0, 30, days` is chosen because it is what the matrix's *own* multilabel
analogue uses — yambda rows 6 and 7 are `LIST_DISTINCT(interactions.event_type, 0, 30, days)`. With the window, `CLASSIFY` compiles to
`multilabel_classification` and `RANK` to `multilabel_ranking`, which are the
two task types the matrix names. Verified across windows: `0, 7, days`,
`0, 30, days` and `0, 365, days` all compile, and the task type depends only on
`CLASSIFY` vs `RANK`.

**This changes the question being asked.** As published the target was "every
element this station has *ever* reported"; as run it is "every element this
station will report in the next 30 days". The second is a genuinely predictive
question and the first is not, so the repair is an improvement — but any
comparison with a published NOAA multilabel baseline has to account for it.

## 3. Row 11's `forecast_length` is PQL, not model plan — NEW, and the matrix is wrong

The matrix says, in prose under the table:

> Forecast horizon is model-plan configuration rather than PQL syntax. Use
> `forecast_length=4` for NOAA weekly TMAX.

On this backend that is inverted in both halves.

**The model-plan field exists and is never read.** `kumoapi/model_plan.py`
declares `forecast_length` on `TrainingTableGenerationPlan` (not
`training_job`), valid for `REGRESSION` and `FORECASTING`, defaulting to
`MissingType.VALUE`. But `kumo_core/training/workflow.py:340` builds the
resolved task with

```python
"forecast_length": request.query.forecast_length,
```

— from the **parsed query**, and `kumo_core/pquery.py:448` sets that as
`int(query.num_forecasts or 1)`. `model_plan.training_table_generation. forecast_length` is not consulted anywhere on this path. (It would not have
arrived anyway; see §7.)

**The PQL clause is `FORECAST n TIMEFRAMES`, and its position matters.** It sits
between the target and `FOR EACH`, where `CLASSIFY TOP k` goes. Putting it after
`FOR EACH` — where a trailing option would naturally go — is a parse error:

```
mismatched input 'FORECAST' expecting <EOF>
```

Measured, on the real graph:

| query                       | task_type         | task_family     | `forecast_length` |
| --------------------------- | ----------------- | --------------- | ----------------: |
| row 11 exactly as published | **`regression`**  | node_prediction |                 1 |
| + `FORECAST 1 TIMEFRAMES`   | `forecasting`     | forecasting     |                 1 |
| + `FORECAST 4 TIMEFRAMES`   | **`forecasting`** | forecasting     |             **4** |
| + `FORECAST 8 TIMEFRAMES`   | `forecasting`     | forecasting     |                 8 |

So **row 11 as published is not a forecasting task** — it silently compiles to
plain temporal regression, which is a different model head
(`ForecastingGNNModule` versus node prediction), a different training table and
a different horizon. `_task_type` in `kumo_core/pquery.py:414` is explicit:

```python
if problem_type == "FORECAST" or forecast_length > 1:
    return "forecasting", None
```

Note the `or`: `FORECAST 1 TIMEFRAMES` *is* forecasting even at length 1,
because the clause sets `problem_type`. The clause, not the number, decides.

Row 11 is run with `FORECAST 4 TIMEFRAMES`; `--forecast-length 0` omits the
clause so the two are comparable on the same tier.

## 4. What the three targets are worth with no model at all — NEW

Rows 2 and 4 each had a structural channel that accounted for most of the
headline score. This fixture has one too, and it is stronger than either of
theirs — but it is not leakage, it is the physics of the problem, and it has to
be quoted beside every number below.

### Rows 5 and 8: the target is last month, repeated

A weather station is a fixed set of instruments. Whatever it recorded in April
it records in May. Measured over the **whole fixture** at the run's anchor
(2019-05-04, `probe_noaa_target.py --stability`), comparing each station's
element set in the 30 days after the anchor against its set in the 30 days
before:

|                                       |           measured |
| ------------------------------------- | -----------------: |
| stations reporting inside the window  |  29,601 of 113,942 |
| stations with no prior 30-day window  |                415 |
| mean labels per station               |           **3.39** |
| **mean recall of "last month's set"** |         **0.9402** |
| mean Jaccard                          |             0.8729 |
| stations whose set is **identical**   | **20,388 (68.9%)** |

So a predictor that simply repeats the previous month recovers 94% of the
target. §8 is what the model adds on top of that: nothing measurable at `tiny`,
and under one point of MAP at `smoke`.

Two more facts from the same probe, both of which bear on how the metrics read:

- **Labels per station inside a window**: min 1, median 3, p90 6, p99 14, max
  **18**. `TOP 25` therefore never binds on an individual entity — it is a
  constraint on the label *vocabulary*, not on any prediction. It is also why
  `recall@25` is uninformative and `precision@25` is pinned near 3.4/25 = 0.14
  by arithmetic. The cut-offs worth reading are 1, 3 and 5.
- **Three quarters of the entity set is dead.** Only 29,601 of 113,942 stations
  report anything in the target window; per tier, 57 of 231 at `tiny` (24.7%)
  and 299 of 1,150 at `smoke` (26.0%). Those entities still get a prediction —
  their true label set is empty, so recall is undefined and they are excluded
  from every rate. Which entities are excluded is itself predictable (a station
  that stopped in 1974 stays stopped), so "is this station still alive?" is a
  second, easier task hiding inside the first, and none of the reported metrics
  separates the two.

### Row 11: the target barely moves week to week

TMAX is in tenths of a degree Celsius, confirmed: over all 431,739,106 TMAX
rows the mean is 176.3 (17.6 °C) and the per-station daily mean has median
165.9 and 5th/95th percentiles 43.2 / 316.8. Contamination is real but small —
1 row at the `-9999` sentinel, 3,578 below −70 °C and 22,500 above +60 °C, so
**0.006%** of TMAX rows are implausible. `quality_flag` is set on 0.51% of them.

The number that makes an MAE readable is how far a weekly mean actually moves.
Measured across the anchor over the whole fixture, restricted to plausible
values:

|                                                       |                   measured |
| ----------------------------------------------------- | -------------------------: |
| stations with a TMAX week on both sides of the anchor |                     10,180 |
| **mean \|change\| in the weekly mean**                | **33.03** tenths (3.30 °C) |
| median \|change\|                                     |     25.51 tenths (2.55 °C) |
| standard deviation of the target itself               |               78.93 tenths |

So persistence scores MAE ≈ 33 and a constant predictor scores roughly 63. A
forecast is only interesting below 33.

## 5. Tiering has to cut on stations, not on rows — NEW

Rows 2–4 hash-sampled the *fact* table because the entity was the fact table.
That is not available here: the target is an aggregate over each station's own
observations, so thinning a station's rows **changes its label**. A station that
reports `WT01` on four days in fifty years loses the class entirely under a 1%
row sample. So a tier is a hash-sampled subset of *stations* carried with all of
their observations, which makes every tier a complete, correct instance of the
published task on fewer entities.

| tier    | stations |  observations | elements present | earliest observation |
| ------- | -------: | ------------: | ---------------: | -------------------- |
| `tiny`  |      231 |     4,862,394 |               61 | 1870-02-01           |
| `smoke` |    1,150 |    27,158,157 |               81 | 1865-04-01           |
| `small` |    5,642 |   141,129,745 |              100 | 1839-01-01           |
| `full`  |  113,942 | 2,895,739,748 |              137 | 1763-01-01           |

Note that **no tier is the published 137-element problem**, and the tiers are
not each other's problem either — the same caveat rows 3 §1 and 4 §3 raise.
Note also that the time span *grows* with the tier, which §6 shows is a second
multiplier on cost, not a neutral property.

`stations` is capped at 113,942, so the full fixture has **fewer entities than
row 2's `smoke` tier had flows**. Going in, that looked like the reason this
block would scale where rows 2–4 did not — row 1's shape, where the entity table
was fixed and only ingest grew. §6 is why the inference is wrong.

## 6. The training table is a full cross product of entities and anchors, and 81% of it is empty — NEW

This is the finding that governs cost, and it inverts the expectation the
dimension-table entity sets up.

Read directly off the generated training table at `tiny`
(`train_tables/*/tt.parquet` and `stats.json` beside it):

|                                        |                            measured |
| -------------------------------------- | ----------------------------------: |
| training-table rows                    |                         **419,958** |
| distinct entities                      |                                 231 |
| **distinct anchors**                   |                           **1,818** |
| anchor cadence                         |        exactly **30 days**, no gaps |
| anchor span                            |             1870-02-01 → 2019-05-01 |
| rows = entities x anchors              | 231 x 1,818 = 419,958 — **exactly** |
| **rows whose target is empty**         |                **340,605 (81.10%)** |
| median anchors a station is active for |              **222 of 1,818 (12%)** |
| mean labels per non-empty row          |             2.90 (median 2, max 23) |
| mean labels over all rows              |                               0.548 |

Every station gets a row at every anchor from the fixture's first observation to
its last, including the 149 years before it existed. There is no filter for
whether the entity was alive, so four fifths of the training signal is the
all-zero multilabel vector, and the BCE loss spends most of its capacity
learning "this station reports nothing".

The consequences are all bad in the same direction:

- **Cost is entities x span, not entities.** A bigger tier buys more stations
  *and* a longer span (§5: `tiny` starts in 1870, `small` in 1839, `full` in
  1763), so anchors grow too. At full scale, 113,942 stations x ~3,118 monthly
  anchors is **≈355 million training rows** for a fixture with 113,942 entities.
- **Ingest is not the constraint, and by a wide margin.** DiskGraph ingested
  `tiny` in **4.78 s** (4,862,625 nodes, 9,724,788 edges, 316 MB read, 109 MB
  written) and `smoke` in **23.97 s** (27.16M observations, 714 MB written) —
  0.98 and 0.88 µs per observation, 22.5 and 26.3 bytes. Extrapolated, the whole
  2.9-billion-row fixture would ingest in **~45 minutes** into **~65–76 GB**,
  comfortably inside the 216 GB available. It is AutoML on a 355-million-row
  training table that puts full scale out of reach, not the graph, and not the
  disk.
- **None of it is configurable.** The anchor cadence comes from
  `training_table_generation.timeframe_step` / `unix_step_size`, and the model
  plan never arrives (§7), so there is no way to coarsen it from the SDK.

### The split follows from the same fact

`stations` has no time column, so the PQL validator warns:

> Entity table stations does not have a "Create date" [...] Training/validation/
> test data split will be done at random, potentially resulting in data leakage.

**The warning is wrong about what happens.** The split is a `DateOffsetSplit`
over the training table's seed times, and it is not random at all:

| tier  | stage |      rows | anchors                                 |
| ----- | ----- | --------: | --------------------------------------- |
| tiny  | train |   419,496 | 1870-02-01 → 2019-03-02 (1,816 anchors) |
|       | val   |   **231** | **2019-04-01 only**                     |
|       | test  |   **231** | **2019-05-01 only**                     |
| smoke | train | 2,156,250 | 1865-04-01 → ... (1,875 anchors)        |
|       | val   | **1,150** | one anchor                              |
|       | test  | **1,150** | one anchor                              |

That is a much healthier split than rows 2, 3 or 4 got — train is the past, test
is the most recent month, and no class is dealt exclusively to one stage, which
is what wrecked rows 3 and 4. It is the one fixture in the series where
`test_*` can be quoted without a paragraph of caveat about the split.

**But it is thin.** The test stage is one anchor and one row per entity, so at
`tiny` the reported `test_ap_macro = 0.8457` is computed over 231 rows of which
only ~57 carry a non-empty label, and at `smoke` `test_ap_macro = 0.9283` over
1,150 of which ~299 do. Both are honest measurements of a small sample. Note
that most of the apparent improvement from `tiny` to `smoke` may be exactly
that: five times the entities in the same single-anchor stage.

One practical note for anyone reading these manifests: `stage_counts` and
`stage_seed_times` **are** present in `published-model/manifests/input.json`,
but `split_kind` reads `None` there and the per-stage *entity* counts appear
only in `train_tables/*/split_stats.json`. `describe_split` reads both, because
the training-table copy is the one that records `DateOffsetSplit` and the
`null` val/test stages that block row 11 (§9).

## 7. The dropped model plan, root cause found — NEW, closes row 3 §7 and row 4 §7

Row 3 §7 measured that `metrics` and `tune_metric` passed to `kumo.Trainer` are
silently dropped and could not isolate where. Row 4 §7 reproduced it on a second
fixture and left it open. It reproduces here on a third fixture and a third pair
of task types, and this run prints the mechanism:

```
metrics          asked=['ap_macro', 'auroc_macro', 'auprc_macro']
                 resolved=[... the nine defaults ...]        -> DROPPED
tune_metric      asked='auroc_macro' resolved='ap_macro'     -> DROPPED

passed to Trainer   dict: {'training_job': {'metrics': [...], 'tune_metric': ...}}
after the request   ModelPlan
seen by kumo_core   {}
```

**The overrides are destroyed at the SDK/backend boundary, before
`kumo_core` sees them.** Three steps:

1. `kumoai_studio/trainer/trainer.py` stores whatever the caller passed — a plain dict
   here — and hands it to `TrainingJobRequest(model_plan=...)`.
1. `TrainingJobRequest` is a **pydantic model** whose `model_plan` field is typed
   `kumoapi.model_plan.ModelPlan`, so pydantic **coerces the dict into a
   `ModelPlan`** on construction.
1. `kumoai_studio/backend/local/training.py::_model_plan_mapping` converts it back by
   looking for `dict`, `model_dump` or `to_dict`. `ModelPlan` is a pydantic
   **dataclass** and has none of the three, so the function falls through to its
   final `return {}`.

`resolve_model_plan` in `kumo_core/training/workflow.py` then merges `{}` over
the defaults, which is why the merge looked correct when row 3 inspected it: the
merge is fine, the payload is already gone. It does not matter whether the
caller passes a dict or a `ModelPlan` — both arrive at `_model_plan_mapping` as
a `ModelPlan`, and both become `{}`.

Reproduced in three lines with no training at all (`explain_plan_boundary`,
printed on every run):

```python
request = TrainingJobRequest({}, pquery_id='probe', graph_snapshot_id='probe',
                             model_plan={'training_job': {'metrics': [...]}})
type(request.model_plan)                   # -> ModelPlan
_model_plan_mapping(request.model_plan)    # -> {}
_model_plan_mapping({'training_job': ...}) # -> the dict, intact
```

### Which runs used which

**The defect was fixed in the working tree partway through this block**, by
another session, at 2026-08-16 23:11 — `_model_plan_mapping` now tries
`as_json()` (the serializer `ModelPlan` actually has) before the three methods
it does not, strips `MissingType.VALUE` sentinels, and raises rather than
returning `{}` on a plan it cannot read. That is an independent confirmation of
the mechanism above, arrived at separately.

It lands in the middle of the run sequence, so **which SDK a run used has to be
stated per run**, and it is, from each run's own `resolved plan` block:

| run                   | started | `_model_plan_mapping` | overrides   | tuned on             |
| --------------------- | ------- | --------------------- | ----------- | -------------------- |
| row 5 `tiny`          | 21:57   | returns `{}`          | **DROPPED** | `ap_macro` (default) |
| row 8 `tiny`          | 22:30   | returns `{}`          | **DROPPED** | `map@10` (default)   |
| row 11 variants       | 23:50 → | returns the plan      | **APPLIED** | `rmse`               |
| row 5 `smoke`         | 01:00   | returns the plan      | **APPLIED** | `auroc_macro`        |
| row 5 `tiny` (re-run) | 05:0x   | returns the plan      | **APPLIED** | `auroc_macro`        |

Two consequences, both stated rather than papered over:

- **The row 5 versus row 8 comparison (§8) is unaffected.** Both ran before the
  fix, both on defaults, on byte-identical training tables. That comparison is
  clean.
- **The `tiny` → `smoke` comparison for row 5 was *not* clean as first run** —
  the tiers differed in tune metric as well as in scale. The row 5 `tiny`
  re-run on the patched SDK exists to separate the two, and §8 quotes it.

Before the fix was found in the tree, this file proposed
`TypeAdapter(ModelPlan).dump_python(model_plan, exclude_defaults=True)`, which
was verified in isolation to recover the payload exactly. The landed fix uses
`as_json()` instead and additionally handles the `'???'` sentinel, which the
`TypeAdapter` route would have forwarded as a literal string — so the landed fix
is the better one, and the proposal is recorded only because it independently
localised the same line.

### What it costs on these three rows — measured

**Nothing measurable**, and that is now a measurement rather than a guess. Row 5
at `tiny` was run both ways on identical data — once with the plan dropped
(tuning on the default `ap_macro`) and once with it applied (tuning on the
requested `auroc_macro`):

|                          | plan dropped | plan applied | difference |
| ------------------------ | -----------: | -----------: | ---------: |
| MAP@5                    |       0.9843 |       0.9834 |    −0.0009 |
| MAP@25                   |       0.9879 |       0.9869 |    −0.0010 |
| NDCG@25                  |       0.9951 |       0.9946 |    −0.0005 |
| macro AUROC              |       0.9806 |       0.9827 |    +0.0021 |
| reported `test_ap_macro` |       0.8457 |       0.8693 |    +0.0236 |
| training time            |    1,753.3 s |    2,904.5 s |   **+66%** |

Every scored difference is inside 57 stations of noise and they do not agree in
sign. The defaults are simply reasonable for all three task types here
(`ap_macro`, `map@10`, `mae`), unlike row 3 where the fallback to `acc` against
an 89% majority selected a degenerate model. The one cost that is real is the
**66% more training time** the applied plan took, which is a different search
trajectory rather than a property of the metric.

So the defect's importance on this block is not what it did to the scores. It is
that **the same boundary would have swallowed `forecast_length`** had it been
passed the way the matrix says to pass it (§3). `forecast_length` is read from
the query instead, so row 11 is recoverable — but a user following the matrix
would have set a model-plan field that was dropped twice over, and would have
trained a plain regression model believing it was a four-step forecast.

## 8. `CLASSIFY` versus `RANK`: same data, same answers, 2.6× the cost — NEW

This is the controlled comparison the block exists for. Rows 5 and 8 are one
query with one word changed, run on the same tier off the same tables against
the same ground truth, and — verified from `stats.json` on both runs — the
**same generated training table**: 419,958 rows, 230 entities, 1,818 anchors,
61 target classes, identical to the row.

### What the word changes inside `kumo_core`

|                               | `CLASSIFY TOP 25`                                      | `RANK TOP 25`                                   |
| ----------------------------- | ------------------------------------------------------ | ----------------------------------------------- |
| task type                     | `multilabel_classification`                            | `multilabel_ranking`                            |
| counts as `is_classification` | **yes**                                                | no                                              |
| counts as `is_ranking`        | no                                                     | **yes**                                         |
| default `tune_metric`         | `ap_macro`                                             | `map@10`                                        |
| default `metrics`             | `ap`/`auroc`/`auprc` × `macro`/`micro`/`per_label` (9) | `map`/`precision`/`recall`/`f1` @ 1/10/100 (12) |
| AUROC / AP reported           | **yes**                                                | **no**                                          |
| NDCG available                | no                                                     | yes, but **not** in the defaults                |
| prediction columns            | `ENTITY, TIMESTAMP, PREDICTED, CLASS, SCORE`           | `ENTITY, TIMESTAMP, CLASS, SCORE`               |
| `PREDICTED` column            | **present** (per-class thresholded boolean)            | **absent**                                      |
| **AutoML trials (measured)**  | **4**                                                  | **8**                                           |
| **training time (measured)**  | **1,753.3 s**                                          | **4,543.6 s**                                   |

The metric sets do not overlap **at all** — `multilabel_classification` reports
nine threshold-free ranking-quality metrics and `multilabel_ranking` reports
twelve cut-off metrics, and there is not one name in common. So the two runs'
own reported numbers cannot be compared to each other, which is precisely why
`score_multilabel` computes one metric set for both from the raw scores.

### What it changes in the answers: nothing measurable

Scored identically, on the same 57 stations (differences in the last column):

| metric      | row 5 `CLASSIFY` | row 8 `RANK` | difference |
| ----------- | ---------------: | -----------: | ---------: |
| MAP@1       |           1.0000 |       1.0000 |          0 |
| MAP@3       |           0.9795 |       0.9737 |    −0.0058 |
| MAP@5       |           0.9843 |       0.9846 |    +0.0003 |
| MAP@25      |           0.9879 |       0.9859 |    −0.0020 |
| NDCG@5      |           0.9913 |       0.9921 |    +0.0008 |
| NDCG@25     |           0.9951 |       0.9940 |    −0.0011 |
| precision@5 |           0.5965 |       0.6000 |    +0.0035 |
| recall@5    |           0.9287 |       0.9313 |    +0.0026 |
| macro AUROC |           0.9806 |       0.9746 |    −0.0060 |

Every gap is under a hundredth, they do not agree in sign, and 57 stations
cannot resolve differences of that size. **The two task types produce the same
model on this fixture**, and `RANK` costs 2.6× the wall time to get there
because its default plan runs eight AutoML trials to `CLASSIFY`'s four.

That is the useful finding for anyone choosing between them: on a target where
both are legal, pick on the **metric set and the output shape** — do you need
`PREDICTED`, do you need AUROC, do you need `@k` — not on expected quality. And
budget for double the search.

### Both tie the no-model baseline

The comparison that matters more than either model is against §4's persistence
ranker — order each station's elements by how often it reported them in the year
before the anchor:

| metric  | persistence |      row 5 |  row 8 |
| ------- | ----------: | ---------: | -----: |
| MAP@1   |      1.0000 |     1.0000 | 1.0000 |
| MAP@3   |  **0.9795** |     0.9795 | 0.9737 |
| MAP@5   |  **0.9873** |     0.9843 | 0.9846 |
| MAP@25  |      0.9858 | **0.9879** | 0.9859 |
| NDCG@5  |  **0.9917** |     0.9913 | 0.9921 |
| NDCG@25 |      0.9907 | **0.9951** | 0.9940 |

At k = 3 and k = 5 persistence is **as good or better than both models**. Only
at k = 25 does either model edge ahead, and there NDCG is nearly saturated for
everything. Against the popularity floor (MAP@5 0.7380, MAP@25 0.8047) both
models are a large, real improvement — so the models did learn per-station
structure — but the structure they learned is the one a `GROUP BY station_id, element` recovers for free.

**At `smoke` the margin appears.** Five times the stations, and row 5 clears
persistence at every cut-off above 1:

| metric  | persistence | row 5 `smoke` |      margin |
| ------- | ----------: | ------------: | ----------: |
| MAP@1   |  **0.9967** |        0.9933 |     −0.0034 |
| MAP@3   |      0.9667 |    **0.9729** | **+0.0062** |
| MAP@5   |      0.9761 |    **0.9832** | **+0.0071** |
| MAP@25  |      0.9781 |    **0.9856** | **+0.0075** |
| NDCG@5  |      0.9886 |    **0.9903** |     +0.0017 |
| NDCG@25 |      0.9906 |    **0.9926** |     +0.0020 |

Consistent in sign across four cut-offs and two metrics, on 299 stations rather
than 57 — so the `tiny` tie looks like a small-sample result rather than a
ceiling. The margin is still **under one point**, and persistence still wins at
k = 1, so the reading does not change much: this fixture's target is
overwhelmingly determined by the station's own recent history, the model
recovers essentially that, and at larger scale it recovers a little more. The
number worth quoting for row 5 is *the margin over persistence*, not MAP@25.

There is a mechanical reason, visible in `published-model/model/encoders.json`:

```
stations     -> {}                                     (no features at all)
observations -> {ELEMENT: Index, VALUE: MaxLogNumerical}
```

`stations` contributes **zero encoded columns** — the published table is one
column wide, so there is no latitude, elevation or country for the
model to generalise *across* stations with. The model's entire input is the
station's own history of `(element, value)` pairs before the anchor, which is
exactly the persistence baseline's input. Given that, the surprising result
would be a large margin, not a small one: the two are reading the same evidence,
and the model's advantage can only come from weighting it better and from
whatever the graph's two-hop neighbourhood (other stations sharing nothing but
the schema) contributes, which is nothing.

So the honest summary is: **the fixture as published gives a model almost no
room above persistence, and the measured margin — zero at 57 stations, under one
point at 299 — is consistent with that.** Adding station attributes (the real
GHCN station file has latitude, longitude, elevation and name) would change the
problem completely, and any future NOAA fixture should carry them.

This is the NOAA analogue of row 2 §3 and row 4 §4, and it is the most extreme
of the three. Row 2's structural channel was worth 0.9256 AUROC against a model
that scored 0.9996 — a real gap. Here the baseline is worth 0.9873 MAP@5 against
a model at 0.9843 (`tiny`) and 0.9761 against 0.9832 (`smoke`): the gap changes
sign with scale and never exceeds a point in either direction. **Do not quote
any of these MAP or NDCG figures without the persistence row beside them** —
alone they read as a near-perfect model, and what they actually measure is a
near-perfectly predictable target.

## 9. Row 11 is blocked: forecasting cannot complete on the local backend — NEW

Row 11 does not run. Not "runs badly" — the forecasting task family fails at two
independent points, and the only shape of the query that completes end to end is
the one that is **not** a forecast.

Five variants, same tier, same tables, same graph, differing only in the
`FORECAST n TIMEFRAMES` clause:

| variant                                            | task type        | val / test split | training           | batch prediction | end to end                     |
| -------------------------------------------------- | ---------------- | ---------------- | ------------------ | ---------------- | ------------------------------ |
| clause omitted                                     | **`regression`** | 18 / 18 rows     | ok, 1,844.8 s      | ok, 0.5 s        | yes — but it is not a forecast |
| `FORECAST 1 TIMEFRAMES`                            | `forecasting`    | 18 / 18 rows     | ok, 1,954.2 s      | **fails**        | **no**                         |
| `FORECAST 2 TIMEFRAMES`                            | `forecasting`    | **null / null**  | **fails, 353.8 s** | —                | **no**                         |
| `FORECAST 3 TIMEFRAMES`                            | `forecasting`    | **null / null**  | **fails, 294.5 s** | —                | **no**                         |
| `FORECAST 4 TIMEFRAMES` (what the matrix asks for) | `forecasting`    | **null / null**  | **fails, 285.1 s** | —                | **no**                         |

**The boundary is exactly at horizon > 1.** A one-step forecast splits normally;
every horizon from 2 upward produces `val_table_stats: null` and
`test_table_stats: null`. The train stage shrinks by one weekly anchor per extra
step — last train anchor 2019-05-07, 04-30, 04-23, 04-16 for horizons 1, 2, 3,
4 — so the horizon is eating the tail of the timeline that the split needs, and
it takes the whole of validation and test with it on the very first step past
one.

### Failure 1 — horizon > 1 leaves no validation or test split

With `FORECAST 4 TIMEFRAMES` every one of the eight AutoML trials dies the same
way:

```
RuntimeError: Early stopping conditioned on metric `val_rmse` which is not
available. Pass in or modify your `EarlyStopping` callback to use any of the
following: `params/learning_rate`, `train_loss`, `train_mae`, `train_mse`,
`train_rmse`
...
RuntimeError: Failed to run any AutoML trial.
```

There is no `val_` metric of any kind, and the training table says why —
`split_stats.json` from that run:

```
train_table_stats  114,713 rows, 73 entities, 1870-02-01 -> 2019-04-16
val_table_stats    null
test_table_stats   null
```

**Both stages are `null`.** Compare the same fixture with the clause omitted:

```
train_table_stats  114,776 rows, 73 entities, 1870-02-01 -> 2019-05-07
val_table_stats    18 rows, 18 entities, 2019-05-14
test_table_stats   18 rows, 18 entities, 2019-05-21
```

The mechanism is visible in the per-timeframe statistics. The anchors are
weekly, and `stats.json` records `num_entities: 0, num_targets: 0` for the last
several — 2019-05-21, 2019-05-14, 2019-05-07 — because a multi-step horizon
needs `7 × n` days of future beyond the anchor and the fixture ends on
2019-06-03. Those trailing anchors are exactly the ones the `DateOffsetSplit`
reserves for validation and test, so for any `n > 1` the split lands entirely
inside the region where no complete label can exist and both stages come back
empty. Training then proceeds with an early-stopping monitor watching a metric
that will never be logged.

**The failure is silent about its cause.** Nothing reports "validation split is
empty"; the message is about an early-stopping callback, eight times, and then a
generic "Failed to run any AutoML trial".

Whether a fixture with a longer tail would avoid it was not tested, and it may
well — the mechanism is "the horizon overruns the end of the data", which more
trailing data would fix. What is measured is that it is **not** a data-*volume*
problem: `tiny` has 4.86M observations and a 149-year span, and a one-step
horizon splits it fine. It is the interaction of horizon length with where the
split is cut, and on this fixture one extra step is enough to break it.

### Failure 2 — batch prediction is not implemented for forecasting

`FORECAST 1 TIMEFRAMES` sidesteps failure 1: the split is the healthy one above
and training completes in 1,954 s. Batch prediction then fails immediately:

```
File "kumo_core/prediction.py", line 63, in generate_prediction_table
    plan_json = compiler.compile(**options)
File "diskgraph/compiler.py", line 276, in compile
    raise DiskgraphConfigError("lag_offsets_secs is required for forecasting tasks")
```

This one is a missing dictionary key, and the compiler's own comment says so:

```python
# diskgraph/compiler.py:274
# We expect the `lag_offsets_secs` to be passed for both training and BP
if not lag_offsets_secs:
    raise DiskgraphConfigError("lag_offsets_secs is required for forecasting tasks")
```

The training path computes them — `kumo_core/pquery.py:487-508` derives
`lag_offsets_secs` and `future_offsets_secs` from the query's timeframe and
`forecast_length`, and it is the **only** place in `kumo_core` that mentions
either. `kumo_core/prediction.py:50-62` builds its own `options` dict:

```python
options = {"query": ..., "timestamps": ..., "task_type": ...,
           "input_timestamp_unit": "nanoseconds", "is_prediction": True,
           "target_stype": ..., "entity_time_col": ...}
if query.task_family == "link_prediction":
    options.update({"lp_prediction_mode": "LHS",
                    "list_distinct_target_is_array": True})
```

There is a special case for `link_prediction` and **none for `forecasting`**, so
`lag_offsets_secs` is never supplied and the compile raises unconditionally.
This is not data-dependent and not tier-dependent: **no forecasting task can
produce batch predictions on this backend**, whatever the fixture. Neither fix
was written or tested here.

### What the regression fallback measures, for what it is worth

The clause-omitted variant is a real run, so its numbers are reported — with the
caveat that it answers a different question (one week ahead, single step) from
the one the matrix asks (four weeks, multi-step).

|                           |                                                                  measured |
| ------------------------- | ------------------------------------------------------------------------: |
| training                  |                                                                 1,844.8 s |
| train / val / test rows   |                                                     114,776 / 18 / **18** |
| prediction output         | `ENTITY, TIMESTAMP, TARGET_PRED` — wide, one row per entity, **231 rows** |
| reported `test_mae`       |                                                           38.70 (3.87 °C) |
| reported `test_rmse`      |                                                                     47.02 |
| reported `test_refit_mae` |                                                                     61.63 |

Scored independently at the anchor 2019-05-27 against the following week:

|                                    |       MAE |      RMSE | stations |
| ---------------------------------- | --------: | --------: | -------: |
| **model**                          | **65.19** | **86.04** |       19 |
| persistence (previous week's mean) | **37.88** |     48.50 |       18 |
| constant (mean of truth)           |     66.39 |     84.75 |       19 |

**The model is worse than persistence and indistinguishable from a constant.**
Two reasons not to make much of that: the evaluation is **19 stations** — only
74 of `tiny`'s 231 stations ever report TMAX and only ~19 are still reporting in
2019 — and the reported `test_mae` is measured on **18 rows**. Neither number
resolves anything. It is reported because it was measured, not because it means
much; a larger tier is the only way to make row 11's regression fallback
informative, and even then it is not row 11.

**Verdict: row 11 is blocked.** The fixture, the graph, the query and the
horizon are all fine — `FORECAST 4 TIMEFRAMES` parses, validates, compiles to
`forecasting` with `forecast_length=4`, and generates a training table. Two
backend defects stop it, one in the split for horizons above 1 and one in the
prediction compiler for every horizon.

## 10. An interrupted ingest wedges the graph permanently — NEW

The first `smoke` attempt was killed mid-ingest (deliberately, to reprioritise).
Every subsequent run of that graph then failed in the first twenty seconds:

```
ingest_runner: attempt 0 failed rc=2 class=fatal
  phase_threads={'context': 5, 'encode': 5, 'pk_map': 5, 'edges_map': 5, 'index': 5}
NonRetryableDiskgraphError: Graph ingest failed for a non-recoverable reason (exit code 2).
  Common causes: a schema mismatch, an invalid column type, a data-validation
  issue, or an OS resource limit such as the open-file cap.
DiskGraph output:
error: invalid value 'context' for '--resume-from-phase <PHASE>'
  [possible values: encode, pk-map, edges-map, index]
```

**The Python side's phase vocabulary is one entry wider than the binary's.**
`kumo_core/diskgraph/runner.py:50` declares

```python
PHASE_ORDER = ("context", "encode", "pk_map", "edges_map", "index")
PHASE_TO_CLI = {"context": "context", "encode": "encode", ...}
```

and `_next_phase_to_run` returns "the lowest phase whose `_phase_complete_<name>`
marker is missing" under the snapshot's `_tmp`. The installed binary accepts only
`encode | pk-map | edges-map | index`, and — measured in the wedged directory —
it **never writes a `_phase_complete_context` marker at all**:

```
_tmp/_phase_complete_encode      present
_tmp/_phase_complete_pk_map      present
_tmp/_phase_complete_context     ABSENT
```

So `context` is always the lowest missing marker, `_next_phase_to_run` always
returns it, and the resume always dies — even here, where `encode` and `pk_map`
had demonstrably finished and the correct resume point was `edges_map`. **The
resume path is unconditionally broken**, not broken only for ingests interrupted
early. Rows 1–4 never saw it because none of their ingests was ever interrupted.

Two things make it worse than a failed retry:

- **It is sticky.** The partial `_tmp` survives, so every later run of that graph
  fails identically. The graph is unusable until the snapshot directory is
  deleted by hand.
- **The error names neither the cause nor the cure.** The wrapper's message
  lists schema mismatch, column type, data validation and file-descriptor
  limits — none of which applies — and the real message is a `clap` argument
  error buried underneath. Recovery is
  `rm -rf <scratch>/diskgraph_snapshots/<graph key>`, which nothing suggests.

The fix is one line: drop `"context"` from `PHASE_ORDER`, or map it to the
binary's first real phase. **Not written or tested here** — the working
directory was cleared instead, and all reported runs used the unmodified
package.

## 11. Carried over from rows 1–4, confirmed again

- **`scikit-learn` is still missing** from the fat wheel, so every run logs
  `Optuna search analysis failed` / `ModuleNotFoundError: No module named 'sklearn'`. Non-fatal, fixed upstream in
  [kumo-ai/kumo#30507](https://github.com/kumo-ai/kumo/pull/30507). Worth
  repeating that this traceback appears in *every* run log, so a wait-loop that
  greps for `Traceback` will report success as failure — row 4 lost a run to
  exactly that, and `run_noaa_tier.sh` waits on the process's exit status
  instead.
- **`validation` metrics come back empty** in the SDK's metric grouping, on all
  five fixtures now. `metrics.validation_metrics` is an empty list even when the
  run logged `val_ap_macro` on every epoch of every trial.
- **`num_entities_predicted` reports rows, not entities** (row 3 §5): 14,091 for
  231 entities × 61 classes on both rows 5 and 8.
- **`_column_stats` corruption does not bite** (row 3 §6). Checked, not
  inherited: none of NOAA's eight column names contains a `ColStatType` token.
- **`Table.infer_metadata` discards stypes set before it**, so `PINNED_STYPES`
  is applied after. Only `STATION_ID` needed pinning (categorical → ID);
  `ELEMENT` inferred as categorical and `VALUE` as numerical unaided, which
  matters because `_task_type` only reaches multilabel through a
  `multicategorical` target derived from a *categorical* column.
- **`SnowflakeConnector(warehouse=...)` is inert locally**; the warehouse is set
  inside the session factory. It still participates in the connector id and so
  in the snapshot's graph key (row 1 §7).
- **`--no-publish-snapshots`** (`kumo.NoopArtifactLocation`) used for every run.
  Irrelevant to disk here — the whole `tiny` index is 120 MB — but kept for
  consistency with rows 1–4.
- **The in-tree SDK on `PYTHONPATH` is not optional.** Without it `kumoai_studio`
  resolves to the fat wheel and the run dies in the first second on
  `AttributeError: module 'kumoai_studio' has no attribute 'NoopArtifactLocation'`.

### New to this block

- **`TOP 25` has no effect on the prediction output.** Every run returns the
  tier's **full element vocabulary** per entity, not 25: 61 classes at `tiny`
  (rows 5 and 8 alike) and **81** at `smoke`. `CLASSIFY TOP k` / `RANK TOP k`
  set `query.top_k`, but the SDK's local prediction path calls
  `trainer.predict` without a `top_k`, and both multilabel branches then fall
  back to the target encoder's full category count. The 20-class truncation row
  3 found applies to `multiclass_classification` only. At full scale this would
  be 137 rows per entity — 15.6 million rows for 113,942 stations, which is
  still small, but the cap the query asked for is simply not applied.
- **The `EVENT_TIME` column arrives as `Dtype.date`, not a timestamp dtype**,
  from a Snowflake `TIMESTAMP_NTZ`. Harmless — the stype is `timestamp` and the
  1763 dates survive — but it is what a hand-written portable graph has to say.
- **Negative event times work end to end.** 1763 to 2019, 256 years, half of it
  before the epoch, through Snowflake, DiskGraph, the training-table generator
  and the split, with no special handling. Only the ad-hoc unit inference in
  `describe_split` needed an `abs()`.

## Reproducing

```bash
source scripts/_env.sh  # resolves the monorepo SDK and root .venv

# One-time: the external stage, then one typed copy of all 2.9B observations
# (250.8 s on KUMO_L).
"$KUMO_SDK_PYTHON" scripts/kaggle/noaa/probe_noaa_fixture.py
"$KUMO_SDK_PYTHON" scripts/kaggle/noaa/build_noaa_tables.py --raw-only

# What the three queries compile to; costs a second and needs no data.
"$KUMO_SDK_PYTHON" scripts/kaggle/noaa/probe_noaa_query.py --variants

# Every matrix claim, and what the targets are worth without a model.
"$KUMO_SDK_PYTHON" scripts/kaggle/noaa/probe_noaa_stats.py --all
"$KUMO_SDK_PYTHON" scripts/kaggle/noaa/probe_noaa_target.py --all

# Build one tier and run one row on it, sampling free disk throughout.
# Scales: tiny | smoke | small | part | half | full (parts per 10,000 of
# station_id). Rows: 5 | 8 | 11. The tier tables are shared by all three rows,
# so the second and third invocations skip the build.
scripts/kaggle/noaa/run_noaa_tier.sh 5 tiny
scripts/kaggle/noaa/run_noaa_tier.sh 8 tiny
scripts/kaggle/noaa/run_noaa_tier.sh 5 smoke

# Row 11's five variants (S9). Only the last completes end to end, and it is
# not a forecast. Horizons 2, 3 and 4 fail in ~5 minutes; horizon 1 trains for
# ~33 minutes and then fails in batch prediction.
scripts/kaggle/noaa/run_noaa_tier.sh 11 tiny                       # FORECAST 4
for n in 3 2 1; do
    scripts/kaggle/noaa/run_noaa_tier.sh 11 tiny --forecast-length $n
done
scripts/kaggle/noaa/run_noaa_tier.sh 11 tiny --forecast-length 0   # regression

# Re-score a finished run without retraining. Batch predictions are durable, so
# this is how every run in the Runs table was put under one metric definition
# even though they were not all run with the same version of the script.
"$KUMO_SDK_PYTHON" scripts/kaggle/noaa/run_noaa_local.py --row 8 --scale tiny \
    --rescore /mnt/localssd/kumo-noaa/artifacts/predictions/<job>/predictions.parquet
```

Recovery, if a run is interrupted during ingest (§10):

```bash
rm -rf /mnt/localssd/kumo-noaa/scratch/diskgraph_snapshots/<graph key>
```

Snowflake objects (account `TLB22710`): stage
`HYU.KAGGLE_FIXTURES.NOAA_GHCN_10GB`; tables `HYU.NOAA.OBSERVATIONS_RAW`
(2,895,739,748 rows) and `HYU.NOAA.STATIONS_RAW` (113,942), plus
`OBSERVATIONS_<TIER>`, `STATIONS_<TIER>` and `STATION_ELEMENTS_<TIER>` per tier.

## Scripts

| file                    | purpose                                                                        |
| ----------------------- | ------------------------------------------------------------------------------ |
| `probe_noaa_fixture.py` | stage the S3 fixture; manifest and inferred schemas                            |
| `probe_noaa_query.py`   | what the three queries compile to, before any data exists (§2, §3)             |
| `probe_noaa_stats.py`   | every matrix claim checked: keys, span, elements, values, columns, part shape  |
| `probe_noaa_target.py`  | what the targets are worth with no model: persistence, coverage, TMAX movement |
| `build_noaa_tables.py`  | one typed raw copy, then station-hash-sampled tiers                            |
| `run_noaa_local.py`     | the end-to-end run for any of rows 5, 8, 11                                    |
| `run_noaa_tier.sh`      | build + run one row on one tier unattended, with disk sampling                 |
