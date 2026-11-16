# NYC FHV trips on the SDK's local backend — matrix rows 9, 12 and 16

One fixture, three matrix rows, run in-process the same way
[row 1](../ethereum/FINDINGS.md), [row 2](../iot23/FINDINGS.md),
[row 3](../bccc/FINDINGS.md), [row 4](../iot23_detailed/FINDINGS.md) and
[rows 5/8/11](../noaa/FINDINGS.md) were: Snowflake catalog reads, DiskGraph
ingest, `kumo_core` AutoML, batch scoring, no Kumo service.

| matrix row | task                     | PQL as published                                                                                      |
| ---------- | ------------------------ | ----------------------------------------------------------------------------------------------------- |
| 9          | regression               | `PREDICT trips.trip_seconds` / `FOR EACH trips.trip_id`                                               |
| 12         | forecasting              | `PREDICT COUNT(trips.*, 0, 1, days)` / `FOR EACH zones.zone_id`, `forecast_length=7`                  |
| 16         | temporal link prediction | `PREDICT LIST_DISTINCT(trips.dropoff_zone_id, 0, 7, days)` / `RANK TOP 20` / `FOR EACH zones.zone_id` |

**Only row 9 runs as published.** Rows 12 and 16 both turn on one thing the
matrix names in a single clause and the backend implements nowhere: the graph
has *two* edges into `zones`, and nothing in PQL can say which one a
`FOR EACH zones.zone_id` query traverses. The choice is made silently by the
**declaration order of the foreign keys** — and getting it wrong is a hard
validation error for row 16 (*"Target trips.dropoff_zone_id creates a loop"*)
and, worse, a silent change of question for row 12, which then forecasts
drop-off demand while reporting itself as pickup demand (S2). Row 12
additionally needs `FORECAST 7 TIMEFRAMES` in the query, which the matrix says
is model-plan configuration and which is not (S3, and NOAA S3 before it);
without it the task compiles to plain `regression`.

With those repairs, the verdict per row:

| row                             | verdict                 | scale               | headline                                                                                                                                                                                                       |
| ------------------------------- | ----------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **9** regression                | **end to end**          | `r9tiny`, `r9smoke` | held-out **R² 0.808 / MAE 194 s** against a `(pickup,dropoff)` median baseline at **R² 0.533 / MAE 317 s** — the first fixture in the series where the model clearly beats its structural baseline (S6)        |
| **12** forecasting              | **blocked, twice over** | `z30`               | the two NOAA defects reproduce on a dense fixture (S5), *and* the regression form that can run predicts a **constant**, because fixed-fanout neighbour sampling erases the very count the target asks for (S5) |
| **16** temporal link prediction | **end to end**          | `z30`, `z90`        | runs, and loses to a `GROUP BY` persistence ranker at every cut-off — because the median zone reaches **215 of 264** possible destinations inside the published window (S7)                                    |

Four defects are new and none is fixture-specific: the anchor edge of a
multi-edge graph is chosen by list order with no warning and no way to see it
(S2); `training_job.metrics` is still ignored after the NOAA fix, for a second
and unrelated reason (S8); **no `timestamp` column is ever a feature** on the
portable path, so hour-of-day and day-of-week are invisible to every model in
this series (S4); and a `COUNT` target on a featureless dimension entity is
unlearnable under the default sampling plan (S5).

## Runs

Everything below is measured on this box (4 vCPU, 15 GB RAM, one L4) with
nothing else running, so the timings are measurements rather than upper bounds.
Two tier families, because the three rows do not share a tiering rule (S9):

| tier      | rule                                 |       trips | span                    |
| --------- | ------------------------------------ | ----------: | ----------------------- |
| `r9tiny`  | `MOD(ABS(HASH(trip_id)), 10000) < 2` |     148,959 | whole 1,399 days        |
| `r9smoke` | `... < 10`                           |     746,514 | whole 1,399 days        |
| `r9small` | `... < 50`                           |   3,728,265 | whole 1,399 days        |
| `z30`     | every trip in days 0–29              |  21,897,074 | 2019-02-01 → 03-02      |
| `z90`     | every trip in days 0–89              |  66,457,571 | 2019-02-01 → 05-01      |
| `z365`    | every trip in days 0–364             | 255,199,632 | 2019-02-01 → 2020-01-31 |
| `zfull`   | every trip                           | 745,287,023 | 1,399 days              |

Row 9 tiers carry a 20% holdout (`trip_seconds` blanked, truth in a side table);
zone tiers carry none, and need none.

|                     | **row 9** `r9tiny` | **row 9** `r9tiny` +dropoff | **row 9** `r9smoke` | **row 12** `z30` F7 | **row 12** `z30` F1 | **row 12** `z30` reg | **row 16** `z30` 7d | **row 16** `z30` 1d | **row 16** `z90` 7d |
| ------------------- | -----------------: | --------------------------: | ------------------: | ------------------: | ------------------: | -------------------: | ------------------: | ------------------: | ------------------: |
| task type           |         regression |                  regression |          regression |         forecasting |         forecasting |           regression |         temporal LP |         temporal LP |         temporal LP |
| end to end          |            **yes** |                     **yes** |             **yes** |              **no** |              **no** |                 yes† |             **yes** |             **yes** |             **yes** |
| AutoML trials       |                  8 |                           8 |                   8 |       8, all failed |                   8 |                    8 |                   8 |                   8 |                   8 |
| DiskGraph ingest    |                4 s |                         4 s |                 8 s |                55 s |              reused |               reused |                55 s |              reused |           **148 s** |
| index on disk       |              10 MB |                       12 MB |               48 MB |              1.4 GB |              reused |               reused |              1.4 GB |              reused |          **4.4 GB** |
| train               |        **316.5 s** |                     397.6 s |       **1,790.8 s** |  **91.5 s to fail** |         **843.7 s** |          **491.4 s** |         **131.6 s** |         **106.4 s** |         **264.5 s** |
| predict             |              2.7 s |                       2.6 s |               8.9 s |                   — |    **fails, 0.6 s** |                1.6 s |               0.9 s |               0.8 s |               0.9 s |
| training-table rows |            119,340 |                     119,320 |             597,144 |           **5,544** |               7,128 |                7,128 |           **1,056** |               7,656 |               3,168 |
| val / test rows     |    11,905 / 12,004 |             11,966 / 11,963 |     60,252 / 59,704 |     **null / null** |           264 / 264 |            264 / 264 |           262 / 263 |           264 / 264 |           262 / 262 |
| prediction rows     |             29,619 |                      29,619 |             149,370 |                   — |                   — |                  264 |              26,400 |              26,400 |              26,400 |
| peak RSS            |             2.1 GB |                      2.1 GB |          **3.2 GB** |              2.8 GB |              2.8 GB |               2.8 GB |              2.8 GB |              2.8 GB |          **6.1 GB** |
| model plan applied  |        partly (S8) |                      partly |              partly |                   — |              partly |               partly |              partly |              partly |              partly |

† Row 12's "regression" column is the published query **without** the
`FORECAST` clause. It completes, and it is not a forecast — and it predicts a
constant (S5).

Not run, and why:

|                                         |                                                                                                    |
| --------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `r9small`                               | ingested cleanly, killed 20 minutes into trial 1 at a measured **~9.0 h** for the full search (S6) |
| `z365` for row 16                       | **OOM-killed** during ingest at 13.5 GB RSS, and wedged the graph for the run after it (S9)        |
| `z365` for row 12                       | never started — it inherited the wedged snapshot (S9)                                              |
| `zfull`                                 | out of reach on 15 GB of RAM: ingest memory extrapolates to the ceiling at ~270M trips (S9)        |
| row 16 with the drop-off edge as anchor | run deliberately; **rejected before training** — that is S2's evidence                             |

## 1. The fixture, checked against the matrix

Second fixture in the series with nothing substantive to correct, and the first
whose key is *genuinely* unique at nine figures.

| claim                                                       | measured                                                                                                                                                                                                |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 745,287,291 rows                                            | **745,287,023** `trips` + **264** `zones` + **4** `providers` = 745,287,291 — the matrix figure is the **sum**, as in every previous row                                                                |
| 18.82 GB stored                                             | 18,824,294,821 B over 155 files (`LIST @stage`), against the manifest's 18,824,293,735                                                                                                                  |
| key `trips.trip_id`                                         | **unique**: 745,287,023 rows, 745,287,023 distinct, 0 null. No dedup needed (contrast rows 1 and 3)                                                                                                     |
| edges `pickup_zone_id`, `dropoff_zone_id` → `zones.zone_id` | **clean both ways**: 0 nulls, 0 orphans on either edge. 263 of 264 zones are ever a pickup, **264 of 264** are ever a drop-off — so exactly one zone can never have a non-empty row 12 or row 16 target |
| `providers`                                                 | 4, all referenced, 0 null foreign keys                                                                                                                                                                  |
| 2019–2022                                                   | **2019-02-01 → 2022-11-30**, 1,399 days, every calendar day present                                                                                                                                     |

One pass over the 150 staged part files into a typed Snowflake table cost
**284.4 s on `KUMO_L`** for all 25 columns and 745 million rows — the cheapest
per-row stage read in the series (0.38 µs/row against NOAA's 0.09 µs on eight
narrow columns and row 3's 30 µs on 325 columns). Everything after it is a scan
of a native table; no tier ever re-reads the stage (row 4 S2).

### The timestamps are microseconds, and there are four of them

All four datetime columns arrive as `NUMBER(38, 0)` microseconds since the
epoch — measured off `part-00000`, whose first `pickup_datetime` is
`1548979286000000` = 2019-02-01, before the expensive pass was started. Nulls
and ordering:

|                     | non-null | note                                            |
| ------------------- | -------: | ----------------------------------------------- |
| `request_datetime`  |   99.99% | **later than pickup in 4,170,636 rows** (0.56%) |
| `on_scene_datetime` |   72.13% | 27.9% null — row 4's `SERVICE` shape, dropped   |
| `pickup_datetime`   |     100% | the event time                                  |
| `dropoff_datetime`  |     100% | **earlier than pickup in 31,497 rows**          |

### Part files are time-contiguous, which is the fourth pattern

Row 1's parts were each a full-history sample, row 2's were one capture each,
NOAA's were globally date-sorted with 5M rows spanning weeks-to-decades. NYC's
are globally sorted *and* strictly contiguous, 5,000,000 rows each:

| part         |      rows | pickup range                        |
| ------------ | --------: | ----------------------------------- |
| `part-00000` | 5,000,000 | 2019-02-01 00:00 → 2019-02-08 02:00 |
| `part-00037` | 5,000,000 | 2019-10-23 20:00 → 2019-10-30 24:00 |
| `part-00075` | 5,000,000 | 2020-12-23 19:00 → 2021-01-07 01:00 |
| `part-00112` | 5,000,000 | 2022-01-17 08:00 → 2022-01-27 11:00 |
| `part-00149` |   287,023 | 2022-11-30 15:00 → 2022-11-30 24:00 |

`trip_id` is `<YYYY-MM>-<sequence>`, so it **sorts by month** and a prefix slice
of either the files or the key is one month of 2019. Confirmed at full scale:
grouping by `LEFT(trip_id, 7)` reproduces the calendar exactly. `MOD(ABS(HASH( trip_id)), 10)` is uniform to four significant figures (74.52M–74.54M per
bucket) and every bucket spans the whole 1,399 days, so the hash is the tier
predicate and the prefix is not.

### The fixture contains the COVID collapse, which matters for row 9's split

| month       |          trips |
| ----------- | -------------: |
| 2019-02     |     20,159,102 |
| 2020-02     |     21,725,100 |
| **2020-03** | **13,392,928** |
| **2020-04** |  **4,312,909** |
| 2020-10     |     13,268,411 |
| 2022-11     |           ~19M |

Volume falls to **20% of trend** in April 2020 and has still not recovered by
the end of the fixture. Two of the four providers also die inside the span:
`HV0002` after 2019-11-18 and `HV0004` after 2021-10-11. Row 9's split is a
`TemporalSplit([0.8, 0.1, 0.1])` cut by row quantile, so train is 2019-02 →
2022-03 and test is 2022-08 → 2022-11 — a different city from most of the
training data (S6).

### Columns

`trips` has 25. Twelve are taken into the graph and thirteen are not:

| column                                                                               |    non-null |    distinct | kept?                                              |
| ------------------------------------------------------------------------------------ | ----------: | ----------: | -------------------------------------------------- |
| `trip_id`                                                                            |        100% | 745,287,023 | primary key                                        |
| `provider_id`                                                                        |        100% |           4 | edge to `providers`                                |
| `pickup_zone_id`                                                                     |        100% |         263 | edge to `zones` — the row 12/16 anchor             |
| `dropoff_zone_id`                                                                    |        100% |         264 | edge to `zones` — the row 16 target                |
| `pickup_datetime`                                                                    |        100% | 115,392,587 | event time                                         |
| `trip_seconds`                                                                       |        100% |      26,026 | the row 9 target                                   |
| `trip_miles`                                                                         |        100% |     122,208 | feature                                            |
| `base_passenger_fare`                                                                |        100% |      61,849 | feature                                            |
| `tips`                                                                               |        100% |      10,178 | feature                                            |
| `driver_pay`                                                                         |        100% |      48,434 | feature                                            |
| `dropoff_datetime`                                                                   |        100% | 114,631,723 | **dropped by default — it is the target (S4)**     |
| `request_datetime`                                                                   |      99.99% | 114,335,421 | dropped with it                                    |
| `on_scene_datetime`                                                                  |      72.13% | 111,525,258 | dropped — 27.9% null                               |
| `originating_base_num`                                                               |      72.59% |          66 | dropped — 27.4% null                               |
| `airport_fee`                                                                        |      44.39% |          23 | dropped — 55.6% null                               |
| `wav_match_flag`                                                                     |      89.53% |           3 | dropped                                            |
| `dispatching_base_num`                                                               |       ~100% |          35 | dropped — a dispatch code, not a measurement       |
| `tolls`, `bcf`, `sales_tax`, `congestion_surcharge`                                  | 100%/99.93% |   3.5K–7.8K | dropped — fare components, redundant with the fare |
| `shared_request_flag`, `shared_match_flag`, `access_a_ride_flag`, `wav_request_flag` |        100% |         2–3 | dropped                                            |

**No stat-token collisions.** Row 3 S6's `_column_stats` defect
(`stat_type.value in key`, matched against the whole key with the column name in
it) is confirmed still present in the installed `kumo_core` and confirmed inert
here — checked on every run over all 25 names plus `zone_id`, including
`dispatching_base_num` and `originating_base_num`, whose `NUM` is not a
`ColStatType` token.

## 2. Two edges into one table, and the anchor is decided by list order — NEW

This is the finding the block turns on, and it is not fixture-specific: it
applies to every matrix row whose `FOR EACH` names a dimension table reachable
by more than one foreign key. Four of the six fixtures in the series ship two
foreign keys into one destination (`transactions.from/to_address → addresses`,
`network_flows.source/destination_ip → hosts` twice, `trips.pickup/ dropoff_zone_id → zones`) and **none of the first five rows ever traversed
one**, because in all of them `FOR EACH` named the fact table. Rows 12 and 16
are the first to anchor on the dimension table.

### There is no PQL syntax for it

`kumopql/grammar/PQLGrammar.g4` has eight rules — `prog`, `problem_spec`,
`problem_type`, `top_k`, `forecast`, `target`, `entity`, `whatif` — and no
`VIA`, `USING` or `THROUGH`. The matrix's instruction to "explicitly select
`pickup_zone_id → zones.zone_id` as the `FOR EACH` anchor" therefore cannot be
written in the query at all. It can only be expressed in the **graph**.

### The mechanism, read off two files and confirmed by compiling both ways

`kumo_core/pquery.py::to_api_graph` emits **one `ColumnKeyGroup` per foreign
key**, so `zones.ZONE_ID` appears in two groups. `kumopql/validator/ join_validator.py::_find_keys` then walks the groups and **assigns**:

```python
for col_group in self.graph.col_groups:
    ...
    if is_keys_group:
        keys = curr_keys        # <- overwrite, not extend
```

so the *last* matching group wins. The "Ambiguous link between tables" error two
hundred lines below (`len(keys) > 1`) is therefore **unreachable for this
shape** — `keys` never holds more than one entry, because each group only ever
contributed one.

Measured, by building the identical graph twice with the two foreign keys
declared in the two orders and compiling the same three queries against each
(`probe_nyc_query.py --fkey-order both`):

| declared last     | row 9                 | row 12 (`COUNT(trips.*)`)         | row 16 (`LIST_DISTINCT(dropoff_zone_id)`)                        |
| ----------------- | --------------------- | --------------------------------- | ---------------------------------------------------------------- |
| `PICKUP_ZONE_ID`  | `regression`, no join | `rhs_key = trips.PICKUP_ZONE_ID`  | **`temporal_link_prediction`**, `rhs_key = trips.PICKUP_ZONE_ID` |
| `DROPOFF_ZONE_ID` | `regression`, no join | `rhs_key = trips.DROPOFF_ZONE_ID` | **REJECTED**                                                     |

The rejection is:

```
Nothing to predict. row 1, column 8: Target trips.DROPOFF_ZONE_ID creates a
loop by pointing to the entity table. A foreign key used in the target
expression must not point to the entity table to define a valid predictive
problem. To fix this, you can change the entity to one of the following:
FOR EACH providers.PROVIDER_ID FOR EACH zones.ZONE_ID.
```

Note the suggestion: it offers `FOR EACH zones.ZONE_ID`, which is what the query
already says. The check at `kumopql/validator/predictive_query_validator.py:429`
fires when `target_ast.rhs_key == target_fqn` — i.e. when the traversal happened
to pick the same edge the target names — so the message describes the *symptom*
of a resolution the user never made and cannot see.

### The two failure modes are of different severity

- **Row 16 fails loudly.** Wrong order, no run. Recoverable in a minute once you
  know the rule.
- **Row 12 fails silently.** Both orders validate, both train, both predict, and
  the two answer *different questions* — pickup demand versus drop-off demand
  per zone. Nothing in the query, the validation output, the resolved plan, the
  split report or the prediction schema distinguishes them. On this fixture the
  two are numerically similar (a zone's daily pickups and drop-offs correlate
  strongly), which makes it worse, not better: a wrong answer that looks right.

So the run script asserts it. `assert_anchor_edge` puts the SDK graph through
the backend's own `_to_portable_graph`, normalizes it, parses the query, and
prints the resolved `target_ast.rhs_key` before training starts:

```
trips fkeys (declaration order): ['PROVIDER_ID', 'DROPOFF_ZONE_ID', 'PICKUP_ZONE_ID']
resolved join key : trips.PICKUP_ZONE_ID
task_type         : temporal_link_prediction (link_prediction)
```

Three lines, no data, and it removes the whole class of error. **Every graph in
the matrix with two edges into one dimension table needs this, and the ordering
rule — *the intended anchor is declared last* — is nowhere documented.**

The alternative fix, an alias table (`dropoff_zones` over the same source, with
`dropoff_zone_id` pointing at it) is deterministic and does not depend on an
ordering bug, but it duplicates the zone node set, moves the RHS lookup
embeddings onto the alias, and departs from the published three-table graph. The
runs reported here use declaration order, so that the graph is the matrix's.

## 3. Row 12 as published is not a forecast — NOAA S3 reproduced exactly

The matrix's prose is the same for NYC as for NOAA:

> Forecast horizon is model-plan configuration rather than PQL syntax. Use
> `forecast_length=4` for NOAA weekly TMAX and `forecast_length=7` for NYC
> daily pickup demand.

Both halves are wrong on this backend, and this block confirms it on a second
fixture and a different aggregation (`COUNT(trips.*)` rather than
`AVG(observations.value WHERE ...)`). Measured, on the real graph:

| query                       | task_type        | task_family     | `forecast_length` |
| --------------------------- | ---------------- | --------------- | ----------------: |
| row 12 exactly as published | **`regression`** | node_prediction |                 1 |
| + `FORECAST 1 TIMEFRAMES`   | `forecasting`    | forecasting     |                 1 |
| + `FORECAST 7 TIMEFRAMES`   | `forecasting`    | forecasting     |             **7** |

`kumo_core/training/workflow.py:340` takes the horizon from
`request.query.forecast_length`, which `kumo_core/pquery.py:448` sets as
`int(query.num_forecasts or 1)` — from the parsed query.
`model_plan.training_table_generation.forecast_length` is never consulted, and
the run reports the disagreement on every row 12 run:

```
forecast_length  query=7 plan=1
```

The clause sits **between** the target and `FOR EACH`. Both of NOAA S3's
observations hold unchanged, so this is carried over rather than new — but it is
worth restating that a user following the matrix would train a plain
single-step regression believing it was a seven-day forecast, and nothing would
tell them otherwise.

## 4. Row 9's target is the difference of two of its own columns — and the backend cannot see it — NEW

`trip_seconds` looks like it is *defined* as `dropoff_datetime - pickup_datetime`. Measured over all 745,287,023 trips:

|                                               |                 measured |
| --------------------------------------------- | -----------------------: |
| `dropoff − pickup = trip_seconds` **exactly** | **632,465,666 (84.86%)** |
| within 1 second                               |     712,151,992 (95.55%) |
| within 60 seconds                             |     735,854,878 (98.73%) |
| mean absolute residual                        |               **3.25 s** |
| median / p99 / max residual                   |      0 / 81.5 / 73,614 s |
| `corr(trip_seconds, dropoff − pickup)`        |               **0.9988** |

Scored as a predictor on the held-out rows, the identity gives **MAE 3.47 s,
R² 0.9965** against a target whose standard deviation is 796 s. So on paper the
published query on the published table is a subtraction.

**In practice it is not, and the reason is a general property of this backend:
no `timestamp` column is ever a feature.** `kumo_core/pquery.py:810`:

```python
feature_columns = [
    column.name for column in table.columns
    if column.name not in excluded
    and column.stype not in {"ID", "timestamp", "unsupported"}
]
```

A `Datetime()` encoder is selected for `stype == "timestamp"` fifty lines above
(`pquery.py:756`) and is **unreachable from the portable path**, because no
timestamp column ever reaches `_encoder_input`. Confirmed from
`published-model/model/encoders.json` on both row 9 runs: the encoder set is

```
zones      -> {}                       (no features at all)
providers  -> {}                       (no features at all)
trips      -> TRIP_MILES: MaxLogNumerical, BASE_PASSENGER_FARE: Numerical,
              TIPS: Numerical, DRIVER_PAY: Numerical
```

— **four numerical columns and nothing else**, in both the default run and the
`--with-dropoff-time` run. Adding `DROPOFF_DATETIME`, `ON_SCENE_DATETIME` and
`REQUEST_DATETIME` to the table adds **zero** encoded columns.

Two consequences, and they pull in opposite directions:

- **Row 9 is not leaking.** The strongest possible feature is present in the
  published table and structurally invisible. That is the good news.
- **Row 9 also cannot see the time of day.** Trip duration in New York is
  strongly a function of hour-of-day and day-of-week — the fixture's own weekly
  cycle runs from 92.3M trips on Mondays to 125.3M on Saturdays — and none of it
  is available to the model. Nor is trip *date*, so the COVID regime change the
  temporal split straddles (S1, S6) is invisible too. **A user who pins a
  timestamp column expecting seasonality features gets silence, not an error.**

The softer proxies survive and are kept, with the caveat stated once here: the
fare columns are computed *after* the trip from its time and distance, so
`corr(trip_seconds, driver_pay) = 0.816`, `corr(·, base_passenger_fare) = 0.780`
and `corr(·, trip_miles) = 0.779`. Row 9 as run is therefore "predict duration
from distance and fare, plus a zone graph" rather than "predict duration at
request time". That is the published query; it is not a forecasting-grade
problem statement, and the S6 baselines are quoted against exactly the same
information.

### What the contrast run measures instead

`--with-dropoff-time` was run at `r9tiny` on identical data specifically to
measure the leak, and since the feature set is provably identical the two runs
differ only in AutoML trajectory:

|               | default | `--with-dropoff-time` | difference |
| ------------- | ------: | --------------------: | ---------: |
| held-out MAE  |  215.45 |                191.26 | **−11.2%** |
| held-out RMSE |  358.52 |                338.94 |      −5.5% |
| held-out R²   |  0.7972 |                0.8188 |     +0.022 |
| training time | 316.5 s |               397.6 s |     +25.6% |

**That 11% is run-to-run AutoML variance on byte-identical features**, not an
effect of the extra columns, and it is the most useful number in this table: it
is the noise floor against which every other paired comparison in this file — and
arguably in the previous five — has to be read. NOAA S7 called a 0.001 MAP
difference "noise on 57 stations"; here a 24-second MAE difference on 29,619
scored rows is also noise, and it is 11% of the metric.

## 5. Row 12 is blocked, and the block is not NOAA's data shape — NEW confirmation

NOAA S9 found forecasting failing at two independent points and left one
question open: whether the horizon-overrun failure was a property of NOAA's thin
tail (a 256-year span whose last observation is 2019-06-03) or of the backend.
NYC FHV is the opposite fixture — 1,399 consecutive days, every one populated,
2,053 trips per zone-day on average, no gaps — and **it fails identically**.

Four variants, same tier (`z30`: 21,897,074 trips over 30 days), same tables,
same graph, differing only in the `FORECAST n TIMEFRAMES` clause:

| variant                              | task type        | train / val / test      | training        | batch prediction | end to end                     |
| ------------------------------------ | ---------------- | ----------------------- | --------------- | ---------------- | ------------------------------ |
| clause omitted                       | **`regression`** | 7,128 / 264 / 264       | ok, **491.4 s** | ok, 1.6 s        | yes — but it is not a forecast |
| `FORECAST 1 TIMEFRAMES`              | `forecasting`    | 7,128 / 264 / 264       | ok, **843.7 s** | **fails**        | **no**                         |
| `FORECAST 7 TIMEFRAMES` (the matrix) | `forecasting`    | **5,544 / null / null** | **fails**       | —                | **no**                         |

### Failure 1 — horizon > 1 leaves no validation or test split

With `FORECAST 7 TIMEFRAMES` every AutoML trial dies the same way:

```
RuntimeError: Early stopping conditioned on metric `val_rmse` which is not
available. Pass in or modify your `EarlyStopping` callback to use any of the
following: `params/learning_rate`, `train_loss`, `train_mae`, `train_mse`,
`train_rmse`
```

and `split_stats.json` says why:

```
train_table_stats  5,544 rows, 265 entities, 2019-02-01 -> 2019-02-21
val_table_stats    null
test_table_stats   null
```

against the one-step variant's

```
train_table_stats  7,128 rows, 265 entities, 2019-02-01 -> 2019-02-27
val_table_stats    264 rows, 265 entities, 2019-02-28
test_table_stats   264 rows, 265 entities, 2019-03-01
```

The generated table has 29 daily anchors either way (`stats.json`
`num_timeframes: 29`, `num_total_rows: 7,656`); the seven-step horizon needs
seven days of future beyond each anchor, so the last eight anchors carry no
complete label, and those are precisely the ones `DateOffsetSplit` reserves for
validation and test. Both stages come back empty and the early-stopping monitor
watches a metric that will never be logged.

**This settles NOAA's open question.** The mechanism is *not* "the fixture ran
out of data": NYC has a dense, gapless daily series and the failure is the same.
It is that the split is cut at the very tail the horizon consumes, with no
interaction between the two — and the trailing anchors are dropped *after* the
split has already been sized. Any horizon above 1 empties val and test on any
fixture whose split is a `DateOffsetSplit`.

**The failure is silent about its cause.** Nothing reports an empty validation
split; the message is about an early-stopping callback, repeated once per trial,
followed by a generic `Failed to run any AutoML trial`.

### Failure 2 — batch prediction is not implemented for forecasting

`FORECAST 1 TIMEFRAMES` sidesteps failure 1 — the split above is healthy and
training completes in 843.7 s with `test_mae = 599.92` — and batch prediction
then fails in 0.6 s:

```
File "kumoai_studio/backend/local/prediction.py", line 351, in _infer
    generated = generate_prediction_table(request)
File "kumo_core/prediction.py", line 63, in generate_prediction_table
    plan_json = compiler.compile(**options)
File "diskgraph/compiler.py", line 276, in compile
    raise DiskgraphConfigError("lag_offsets_secs is required for forecasting tasks")
```

Reproduced without training anything as well (`probe_nyc_query.py`), on both
`FORECAST 1` and `FORECAST 7`, in both training and prediction compile modes.

`kumo_core/prediction.py:50-62` builds the compile options and special-cases
exactly one task family:

```python
if query.task_family == "link_prediction":
    options.update({"lp_prediction_mode": "LHS",
                    "list_distinct_target_is_array": True})
```

There is no `forecasting` branch, so `lag_offsets_secs` and
`future_offsets_secs` — which `kumo_core/pquery.py::_forecast_options` derives
for the *training* path and wires into `TrainTableRequest` — are never put into
`PredictionTableRequest`, and `diskgraph/compiler.py` raises before it reaches
the `is_prediction` branch. This is not data-dependent, not tier-dependent and
not horizon-dependent: **no forecasting task can produce batch predictions on
this backend, on any fixture.**

**Verdict: row 12 is blocked.** The query parses, validates, compiles to
`forecasting` with `forecast_length=7`, resolves the correct anchor edge and
generates a training table. Two backend defects stop it, one in the split for
horizons above 1 and one in the prediction compiler for every horizon. Both were
first found on NOAA; this block confirms neither is specific to that fixture.

### What the regression fallback measures, and why it is worse than nothing

The clause-omitted variant is a real run and completes end to end, so its
numbers are reported — with the caveat that it answers a different question
(one day ahead, single step) from the one the matrix asks (seven days,
multi-step).

|                      |                                                         measured |
| -------------------- | ---------------------------------------------------------------: |
| training             |                                            **491.4 s**, 8 trials |
| train / val / test   | 7,128 / 264 / **264** rows; val and test are **one anchor each** |
| prediction output    |        `ENTITY, TIMESTAMP, TARGET_PRED` — 264 rows, one per zone |
| reported `test_mae`  |                                                         2,221.20 |
| reported `test_rmse` |                                                         3,118.13 |

Scored independently at the anchor 2019-03-01 against the 900,381 trips actually
picked up the next day:

|                            |          MAE |         RMSE |       WAPE |          R² |
| -------------------------- | -----------: | -----------: | ---------: | ----------: |
| **model**                  | **2,423.55** | **3,558.91** | **0.7106** | **−0.1745** |
| persistence (previous day) |   **521.36** |     1,124.73 |     0.1529 |  **0.8827** |
| 4-week mean                |       798.52 |     1,300.84 |     0.2341 |      0.8431 |
| constant (mean of truth)   |     2,561.77 |     3,283.89 |     0.7511 |      0.0000 |

**The model is worse than a constant.** It is not merely weak — it *is* a
constant: the 264 predictions have standard deviation **132** against a truth
standard deviation of **3,290**, and 263 of them lie between 2,029.08 and
2,029.11. Twelve distinct values to two decimal places.

### Why a `COUNT` target on a dimension-table entity is unlearnable here — NEW

This is a structural result, not a tuning failure, and it applies to row 12 at
any tier and any horizon.

- `zones` contributes **zero encoded features** — `encoders.json` reads
  `zones: {}`, exactly as NOAA's `stations` did. The published table is one
  column wide. So everything the model knows about a zone comes through the
  edge.
- The target is `COUNT(trips.*, 0, 1, days)` — literally *how many* trips the
  zone has.
- Neighbour sampling is `num_neighbors = [hop1: 16, hop2: 16]`, fixed fanout.

A zone with 20,000 trips on the anchor day and a zone with 20 both present
**16 sampled trips** to the encoder. The one quantity the task asks for is the
one quantity fixed-fanout sampling erases. There is no feature, anywhere in the
subgraph, that is monotone in the count, so the best the network can do is
predict the global mean — which is precisely what it does.

`sample_from_entity_table` and `max_target_neighbors_per_entity` both resolve to
the `'???'` sentinel (unset) and `adaptive_sampling` is `False`, so nothing in
the default plan compensates. A degree feature on the entity table, an adaptive
fanout, or a much larger `num_neighbors` would all change this; none is reachable
from the published fixture and the default plan.

**So row 12 is doubly blocked**: the forecasting form cannot train (horizon > 1)
or predict (any horizon), and the regression form that *can* run is answering a
question its architecture cannot represent. The number to quote for row 12 is
the persistence baseline — MAE 521 on a mean of 3,411, WAPE 0.153 — and the
observation that the model does not reach it.

## 6. Row 9, the series' first regression row — NEW

Row 9 is the only one of the three that runs as published, and it is the first
regression task in the six-fixture series, so its defaults are recorded in full.

|                    | measured                                                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| task type / family | `regression` / `node_prediction`, `query_type = static`                                                               |
| split              | `TemporalSplit([0.8, 0.1, 0.1])` — `kumo_core/pquery.py:513` picks it because the entity table has a time column      |
| default metrics    | `mae`, `mse`, `rmse`; `tune_metric = mae`                                                                             |
| loss               | Huber                                                                                                                 |
| target scaling     | **none** — `NumericalTargetEncoder` with `scaler=None`, an identity. No winsorization, no log transform, no plan knob |
| AutoML trials      | 8, `max_epochs = 12`                                                                                                  |
| prediction output  | wide: `ENTITY`, `TARGET_PRED`, one row per entity                                                                     |
| holdout            | **required**, 20%, `WHERE trip_seconds IS NULL`                                                                       |
| encoded features   | four numericals; `zones` and `providers` contribute **zero** (S4)                                                     |

### Results

Scored on the held-out 20% — rows whose `trip_seconds` was blanked in the graph
table, with truth in `TRIPS_<TIER>_TRUTH` — against four no-model baselines
computed on the *labeled* rows only:

|                      |    `r9tiny` | `r9tiny` + dropoff time |     `r9smoke` |
| -------------------- | ----------: | ----------------------: | ------------: |
| trips in the tier    |     148,959 |                 148,959 |       746,514 |
| held out / scored    |      29,619 |                  29,619 |       149,370 |
| training-table rows  |     119,340 |                 119,320 |       597,144 |
| AutoML trials        |           8 |                       8 |             8 |
| **train**            | **316.5 s** |                 397.6 s | **1,790.8 s** |
| predict              |       2.7 s |                   2.6 s |         8.9 s |
| DiskGraph index      |       10 MB |                   12 MB |     **48 MB** |
| train tables on disk |      166 MB |                  166 MB |    **823 MB** |
| peak RSS             |      2.1 GB |                  2.1 GB |    **3.2 GB** |
| reported `test_mae`  |      187.97 |                  204.44 |        167.55 |
| reported `test_rmse` |      411.91 |                  394.25 |        354.61 |
| **held-out MAE**     |  **215.45** |              **191.26** |    **194.37** |
| **held-out RMSE**    |      358.52 |                  338.94 |        350.70 |
| **held-out WAPE**    |      0.1927 |                  0.1711 |        0.1738 |
| **held-out R²**      |  **0.7972** |              **0.8188** |    **0.8082** |

and the baselines, on the same rows:

| baseline                                 | `r9tiny` MAE | `r9smoke` MAE | `r9smoke` R² |
| ---------------------------------------- | -----------: | ------------: | -----------: |
| constant (labeled mean)                  |       570.35 |        569.63 |      −0.0000 |
| **`(pickup_zone, dropoff_zone)` median** |   **351.93** |    **316.91** |   **0.5333** |
| `trip_miles` × mean pace                 |       659.37 |        656.56 |      −1.9984 |
| *identity `dropoff − pickup`*            |       *3.47* |        *3.31* |     *0.9979* |

**Read the zone-pair row.** It is a two-column `GROUP BY` that uses exactly the
structure the GNN is being paid to learn — the pickup zone, the drop-off zone,
and nothing else — and it is the number row 9 has to beat. It does, and by a
wide margin: **MAE 194 against 317, R² 0.808 against 0.533** at `r9smoke`.
That is the **first fixture in the series where the model clearly beats its
structural baseline.** Rows 2 and 4 had a structural channel worth most of the
score; NOAA's persistence baseline tied or beat the model at every cut-off. Here
the graph plus four fare/distance columns recover something a group-by does not.

The whole-fixture version of the same baseline, computed in SQL over all
745,287,023 trips with the same 20% holdout (`probe_nyc_target.py --duration`),
is **MAE 291.92 / RMSE 497.43** with only 238 held-out trips landing on a
zone pair never seen in training. So the baseline is stable across scales and
the tier estimates of it are honest.

### Scale bought nothing measurable, and the noise floor is why

`r9tiny` → `r9smoke` is **5.00× the training rows for 5.66× the time**, an
exponent of **1.08** — mildly superlinear, and much better behaved than rows
2–4's 0.22–0.93 spread, probably because this fixture's per-row cost is nearly
constant (four encoded columns, no wide feature table).

Held-out MAE went 215.45 → 194.37, which looks like a 10% improvement for 5× the
data. **It is not distinguishable from noise**: the two `r9tiny` runs, on
byte-identical features (S4), differ by 215.45 → 191.26, and `r9smoke`'s 194.37
sits inside that interval. The honest statement is that **five times the trips
did not measurably improve held-out accuracy on this target**, which is
consistent with S4's picture — the model has four numeric columns and a zone
graph, and a quarter of a million trips already saturates what that can express.

### `r9small` was attempted and abandoned

`r9small` (3,728,265 trips, 2,982,353 training rows, 5× `r9smoke`) ingested
cleanly — 242 MB of index, 4.0 GB of training table, both exactly on the
per-row constants above — and was **killed 20 minutes into its first AutoML
trial**, because the measured rate makes it a nine-hour run:

|                      |            `r9smoke` |                                 `r9small` |
| -------------------- | -------------------: | ----------------------------------------: |
| training rows        |              597,144 |                                 2,982,353 |
| steps per epoch      |                  933 | **2,000** (the `max_steps_per_epoch` cap) |
| measured rate        |             ~37 it/s |                             **5.90 it/s** |
| seconds per epoch    |                  ~25 |                                  **~339** |
| 12 epochs × 8 trials | 1,790.8 s (measured) |                 **~9.0 h** (extrapolated) |

The interesting part is *where* the 6× per-step slowdown comes from, since the
batch size and the model are the same: the graph is 5× larger, so each
neighbour sample walks a bigger index. Note also that `max_steps_per_epoch = 2000` binds at `r9small` and not at `r9smoke`, so past this point an epoch stops
being a pass over the data and becomes a fixed 2,000-step budget — which means
larger tiers see a *smaller fraction* of their own training table per epoch,
and the 1.08 exponent measured below does not extend past here.

### Full scale is out of reach, and the binding constraint is the training table

Not the index, and not AutoML. Measured on both tiers, the DiskGraph index is
**64 bytes per trip** (48 MB for 746,514; 1.4 GB for 21,897,074 at `z30`), so
all 745,287,023 trips would index into about **48 GB** against 213 GB free —
comfortable. The **generated training table** is **1.38 KB per row** (823 MB for
597,144), and row 9's training table is one row per labeled trip, so full scale
is 596 million rows and roughly **800 GB**. That is 3.7× the free disk.

The disk ceiling is therefore about **160 million training rows ≈ 200 million
trips ≈ 27% of the fixture**, before considering that 1.08-exponent time. At
1,790.8 s for 597,144 rows, 160 million rows extrapolates to on the order of
**200 hours**, so time binds long before disk does. `r9small` (3.7M trips) is
the largest tier attempted here; anything past `r9big` (74.5M trips, ~80 GB of
training table) would need a machine with more than one L4.

## 7. Row 16, the series' first link prediction row — and the target is not sparse — NEW

Row 16 runs end to end, at every tier tried, and produces a model that is
**worse than a single global ranking**. The reason is not the model: it is that
the task as published is close to degenerate on this fixture, and one SQL query
says so before any training.

### The candidate set is 264 and a zone reaches most of it every day

`LIST_DISTINCT(trips.dropoff_zone_id, 0, 7, days)` asks, for each zone, which
zones its trips will end in over the next week. Measured over the whole fixture
at 2022-06-01 (`probe_nyc_target.py --destinations`):

| window | destinations per pickup zone (p5 / median / p95 / max) |
| ------ | ------------------------------------------------------ |
| 1 day  | 35 / **138** / 206 / 239                               |
| 3 days | 57 / **188** / 237 / 250                               |
| 7 days | 90 / **215** / 249 / 256                               |

Out of 264 possible destinations. **The median zone reaches 81% of the entire
candidate universe inside the published window, and 52% inside a single day.**

That is the opposite of every link-prediction fixture the matrix is modelled on.
Yambda's analogue (`LIST_DISTINCT(interactions.item_id, 0, 30, days)`,
`RANK TOP 100`) has a candidate set in the millions and a user touches tens of
items; here the candidate set is 264 and an entity touches 215 of them. `RANK TOP 20` against a 215-element truth set is not "find the needles", it is "name
any 20 of the 215 correct answers", and precision@20 is near 1 for anything that
is not adversarial. It follows arithmetically that recall@20 cannot exceed
20/215 ≈ 0.093 — and the measured value is 0.098.

The stability of that set is correspondingly high, NOAA S4's measurement
transposed to zone pairs, across 2022-06-01 over the whole fixture:

|                                            |   measured |
| ------------------------------------------ | ---------: |
| zones with destinations on both sides      |        260 |
| mean destinations in the following week    |      201.4 |
| **mean recall of the previous week's set** | **0.9069** |
| mean Jaccard                               |     0.8309 |

Drop-off traffic is also only mildly concentrated — the busiest destination zone
(265) takes 3.49% of all drop-offs, the top twelve take 17.7% — so a popularity
ranking is not trivially degenerate either; it is simply *good*, because almost
every pair occurs.

### Results at `z30`

|                   | measured                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| task type         | `temporal_link_prediction` (`link_prediction`)                                                  |
| anchor edge       | `trips.PICKUP_ZONE_ID`, asserted (S2)                                                           |
| training table    | **1,056 rows** = 264 zones × 4 weekly anchors                                                   |
| split             | train 524 rows / 2 anchors (2019-02-01, 02-08); val 262 / **1 anchor**; test 263 / **1 anchor** |
| train             | **131.6 s**, 8 trials                                                                           |
| predict           | 0.9 s                                                                                           |
| DiskGraph index   | 1.4 GB for 21,897,074 trips (**64 B/trip**)                                                     |
| prediction output | long: `ENTITY, CLASS, SCORE, TIMESTAMP`, **no rank column**                                     |
| prediction rows   | **26,400** = 264 × **100**, when `TOP 20` was asked for                                         |
| holdout           | **not needed** — the LP prediction plan is `pass_through` with no filter                        |

Reported by the run: `test_map@1 = 0.9848`, `test_map@10 = 0.9031`,
`test_map@100 = 0.8383`, `test_precision@100 = 0.8740`,
`test_recall@100 = 0.4638` — the **twelve defaults**, not the four asked for
(S8).

Scored independently against the destinations actually reached in the seven days
after the anchor (2019-02-23), 262 of 264 zones having a non-empty target:

|                  |   k |     prec@k |   recall@k |    **MAP@k** |   **NDCG@k** |
| ---------------- | --: | ---------: | ---------: | -----------: | -----------: |
| **row 16 model** |   1 |     0.9885 |     0.0066 |   **0.9885** |   **0.9885** |
|                  |   3 |     0.9466 |     0.0164 |   **0.9427** |   **0.9548** |
|                  |   5 |     0.9321 |     0.0260 |   **0.9244** |   **0.9425** |
|                  |  10 |     0.9153 |     0.0502 |   **0.8978** |   **0.9278** |
|                  |  20 |     0.9071 |     0.0977 |   **0.8823** |   **0.9184** |
| *popularity*     |   1 |     0.9733 |     0.0057 |     *0.9733* |     *0.9733* |
|                  |   5 |     0.9527 |     0.0266 |     *0.9476* |     *0.9587* |
|                  |  20 |     0.9286 |     0.1036 |     *0.9106* |     *0.9375* |
| *persistence*    |   1 |     0.9962 |     0.0086 |     *0.9962* |     *0.9962* |
|                  |   5 |     0.9936 |     0.0321 |     *0.9912* |     *0.9947* |
|                  |  20 | **0.9881** | **0.1172** | ***0.9844*** | ***0.9906*** |

**The model loses to both baselines at every cut-off above 1**, and to
persistence at every cut-off including 1. Persistence here is "rank this zone's
destinations by how many trips went there in the previous seven days", which is
a `GROUP BY pickup_zone_id, dropoff_zone_id` — one query, no model, MAP@20
0.9844 against the model's 0.8823.

Two honest qualifications, in opposite directions:

- **The split is one anchor.** `z30` gives four weekly anchors: two to train,
  one to validate, one to test. 1,056 training rows is not enough to fit
  anything, and the model's own `test_map@1` (0.9848) is measured on 263 rows.
  The `z365` run below is what tests whether more anchors close the gap.
- **The metric is saturated for everyone.** MAP@20 runs 0.88–0.98 across a
  model, a popularity ranking and a persistence ranking; nothing in that range
  is informative about ranking quality. Recall@k is the only unsaturated
  column, and it is pinned by arithmetic (k/215). **Do not quote row 16's MAP or
  NDCG without the persistence row.**

The useful reading is not "the model is bad" but **"this fixture is the wrong
shape for the task the matrix assigns it"**. A temporal link-prediction fixture
needs a candidate set much larger than the per-entity truth set; NYC's zone
graph is nearly complete. Making it a real ranking problem would need either a
much finer destination granularity (the raw TLC data has ~265 zones and no finer
key) or a much shorter window — and S7's window table shows even one day leaves
138 of 264.

### More anchors do not close the gap, and a shorter window changes the verdict

Two contrasts, both run:

**`z90` — 3× the trips and 3× the anchors** (66,457,571 trips, 12 weekly
anchors, 3,168 training rows, train 264.5 s):

| metric  | model `z30` | model `z90` | popularity `z90` | persistence `z90` |
| ------- | ----------: | ----------: | ---------------: | ----------------: |
| MAP@1   |      0.9885 |      0.9734 |           0.9658 |        **0.9885** |
| MAP@5   |      0.9244 |      0.9282 |           0.9400 |        **0.9864** |
| MAP@20  |      0.8823 |      0.8884 |           0.9116 |        **0.9791** |
| NDCG@20 |      0.9184 |      0.9235 |           0.9373 |        **0.9832** |

Tripling the data moves MAP@20 by **+0.006** and leaves the model still below
popularity and 0.09 below persistence. This is not a small-sample result.

**A 1-day window** (`--window-days 1`, same `z30` tier, so the only change is
the target's horizon; truth median falls from 215 destinations to 153):

| metric |      model | popularity | persistence |
| ------ | ---------: | ---------: | ----------: |
| MAP@3  | **0.9505** |     0.8589 |    *0.9903* |
| MAP@5  | **0.9124** |     0.8271 |    *0.9864* |
| MAP@10 | **0.8494** |     0.7999 |    *0.9757* |
| MAP@20 | **0.7936** |     0.7620 |    *0.9590* |

Narrowing the window desaturates the task — every number drops — and the model
then **clears the popularity floor at every cut-off above 1**, by 0.03–0.09
MAP, which it never does at seven days. It is still nowhere near persistence.

So the published seven-day window is the worst of both: saturated enough that a
global ranking beats the model, and not sparse enough for the ranking to mean
anything. The one-day variant is a better instance of the same task and would
be the one to publish.

## 8. The model plan now arrives — and `metrics` is still ignored, for a second, different reason — NEW

NOAA S7 found and named the boundary that swallowed every model-plan override on
rows 3, 4, 5 and 8: `TrainingJobRequest` coerces the caller's dict into a
`ModelPlan` pydantic *dataclass*, which has no `dict` / `model_dump` /
`to_dict`, so `kumoai_studio/backend/local/training.py::_model_plan_mapping` fell
through to `return {}`. That fix is in the working tree and **it works**. Every
run in this block prints the three-line reproduction, and on all of them:

```
passed to Trainer   dict: {'training_job': {'metrics': [...], 'tune_metric': 'rmse'}}
after the request   ModelPlan
seen by kumo_core   {'training_job': {'metrics': [...], 'tune_metric': 'rmse', ...}}
```

The plan survives, and `resolved-model-plan.json` records it verbatim:

```
metrics      asked=['mae','mse','rmse','r2','wape'] resolved=['mae','mse','rmse','r2','wape']  -> APPLIED
tune_metric  asked='rmse'                           resolved='rmse'                            -> APPLIED
```

**And `r2` and `wape` are never computed.** `published-model/metrics.json` from
the same run holds `loss`, `mae`, `mse`, `rmse` and their `refit_` twins and
nothing else — which is exactly `default_metrics(TaskType.REGRESSION) = ['mae','mse','rmse']`. Row 16 makes it unambiguous: it asked for four LP metrics
and the run reported **all twelve defaults** plus `locality_score`, i.e. a
superset of the request including metrics never asked for.

The cause is a second, independent break, in a different file from NOAA's, and
it is a **section-name mismatch**. `kumo_core/training/workflow.py:308`:

```python
task_options = request.planning_options.get("task", {})
...
"metrics": (list(task_options["metrics"]) if "metrics" in task_options else None),
```

It reads the metric list from a **`task`** section of the model plan.
`kumoapi.model_plan.ModelPlan` has no `task` section — metrics live under
`training_job` — so `task_options` is always `{}`, `metrics` is always `None`,
and `ResolvedTask` falls back to `default_metrics(task_type)`
(`kumo_core/resolved_task.py:236`).

`tune_metric` takes a *different* route and does arrive:
`kumo_core/automl/auto_trainer.py:268` reads
`space.training_job.tune_metric`. But two lines at `auto_trainer.py:1352`
guard it:

```python
if model_plan.training_job.tune_metric not in task._metrics:
    model_plan.training_job.tune_metric = task._metrics[0]
```

and `task._metrics` is the defaulted list. So **`tune_metric` is honoured if and
only if it happens to be one of the task's default metrics, and is silently
rewritten to the first default otherwise.** Asking to tune on `r2` for a
regression task would yield a model tuned on `mae`, with `r2` in the resolved
plan and no warning anywhere.

Net effect, stated precisely because NOAA S7's conclusion needs amending:

| override                                 | before the NOAA fix | after it, measured here                      |
| ---------------------------------------- | ------------------- | -------------------------------------------- |
| `training_job.tune_metric`               | dropped             | **applied**, but only within the default set |
| `training_job.metrics`                   | dropped             | **still dropped**, at `workflow.py:308`      |
| `training_job.*` (other fields)          | dropped             | applied                                      |
| `neighbor_sampling`, `column_processing` | dropped             | applied                                      |

So the tuning-trajectory difference NOAA S7 measured (its `tiny` re-run costing
66% more training time on the same data) was real — `tune_metric` does land —
but its statement that overrides now arrive is only three-quarters true. The run
script prints the verdict from the *reported metric names* rather than from the
resolved plan, because on this backend those two disagree.

## 9. Cost, and what each row's constraint actually is — NEW

The three rows have three different binding constraints, which is why the tier
ladder is split in two (`build_nyc_tables.py`).

|                    | row 9                                                                                                | rows 12 and 16                                                                                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| entity             | `trips`, 745M rows                                                                                   | `zones`, 264 rows                                                                                                                                                     |
| a tier is          | a uniform hash sample of `trip_id` over the whole span                                               | every trip inside a contiguous window                                                                                                                                 |
| why                | `trip_seconds` is a static column, so thinning removes entities without changing any surviving label | both targets are aggregates over a zone's trips, so a row sample **divides the label** — a 1% sample divides every daily `COUNT` by 100 and deletes most destinations |
| training table     | one row per labeled trip — **millions**                                                              | entities × anchors — **thousands**                                                                                                                                    |
| binding constraint | **training-table disk**, then time                                                                   | **DiskGraph ingest**                                                                                                                                                  |

The zone tiers cannot follow NOAA's rule (sample the *entity* table) because 264
zones is already the smallest the entity set gets and the graph contract wants
all of them. Time is the only axis left, and it is the right one anyway: for a
dimension-table entity the training table is entities × anchors, so a longer
window buys anchors, which is what these rows are short of.

### Measured constants

|                                                    |                                                     measured |
| -------------------------------------------------- | -----------------------------------------------------------: |
| stage → typed Snowflake table, 745M rows × 25 cols |                                      **284.4 s** on `KUMO_L` |
| tier build off the raw table                       |                                                   **5–20 s** |
| **DiskGraph index**                                | **64 bytes per trip** (48 MB / 746,514; 1.4 GB / 21,897,074) |
| DiskGraph ingest                                   |                   **~2.5 µs per trip** (55 s for 21,897,074) |
| **generated training table**                       |                       **1.38 KB per row** (823 MB / 597,144) |
| peak RSS                                           |                      2.1 GB at `r9tiny`, 3.2 GB at `r9smoke` |

### The real ceiling is DiskGraph ingest **memory**, not disk — NEW

The disk arithmetic says rows 12 and 16 should reach `zfull`: 745,287,023 trips
index into about **48 GB** against 213 GB free, in roughly 31 minutes. **They do
not.** `z365` — 255,199,632 trips, one year — was attempted for row 16 and the
ingest binary was **OOM-killed**:

```
oom-kill: ... task=ingest, pid=817528
Out of memory: Killed process 817528 (ingest)
    total-vm:19136352kB, anon-rss:13558432kB
```

sampled at **13.08 GB RSS** one minute before the kill, on a 15 GB machine. It
died in the **Index** phase, 74% of the way through `trips`
(191,388,736 of 255,199,632), after the on-disk index had already reached 17 GB.
`kumo_core/diskgraph/runner.py` retried it eight times, each retry was killed the
same way, and the run ended with

```
Error: DiskGraph ingest exhausted 8 worker retries
```

So ingest memory scales with the table and is **not** streamed within the index
phase. The measured points are 21.9M trips (fine, ~2 GB) and 255M trips (13.5 GB
and killed); linearly that puts the ceiling near **270 million trips**, i.e.
about **36% of the fixture**, on 15 GB of RAM. `z90` (66,457,571 trips) is the
largest zone tier this block completed.

**This also reproduces NOAA S10 on a second fixture, and from a second cause.**
The OOM leaves a partial `_tmp` under the snapshot directory, and the *next* run
against that graph key dies in twenty seconds with

```
error: invalid value 'context' for '--resume-from-phase <PHASE>'
  [possible values: encode, pk-map, edges-map, index]
```

— `PHASE_ORDER` in `kumo_core/diskgraph/runner.py:50` includes a `context` phase
the binary does not accept and never writes a marker for, so the lowest missing
marker is always `context` and every resume fails. NOAA reached this by
interrupting an ingest deliberately; here it arrived unbidden, via the OOM, and
took out the *following* queued run (row 12 at `z365`) as collateral. Recovery is
still `rm -rf <scratch>/diskgraph_snapshots/<graph key>`, which nothing suggests.

- **Row 9 cannot.** Its training table is one row per labeled trip, so full
  scale is ~596 million rows and ~**800 GB** — 3.7× the free disk — and the
  1.08 time exponent puts 160 million rows (the disk ceiling) at order **200
  hours**. Stated as a ceiling rather than an extrapolation: rows 2 and 4 both
  retracted two-point fits, and one exponent from two tiers is one exponent from
  two tiers.

Disk never came close to binding at the scales actually run: the whole
`/mnt/localssd` footprint peaked at **3.6 GiB** of 217 available across every
run in this block.

## 10. Carried over from rows 1–5, 8 and 11, confirmed again

- **`scikit-learn` is still missing** from the fat wheel, so every run logs
  `Optuna search analysis failed` / `ModuleNotFoundError: No module named 'sklearn'`. Non-fatal, fixed upstream in
  [kumo-ai/kumo#30507](https://github.com/kumo-ai/kumo/pull/30507). It appears
  in *every* run log, so a wait-loop that greps for `Traceback` reports success
  as failure — row 4 lost a run to exactly that, and `run_nyc_tier.sh` waits on
  the process's exit status instead.
- **`validation` metrics come back empty** in the SDK's metric grouping, on all
  six fixtures now. `metrics.validation_metrics` is an empty list even when the
  run logged `val_mae` on every epoch of every trial.
- **`num_entities_predicted` reports rows, not entities** (row 3 S5): 26,400 for
  264 zones × 100 candidates on row 16. For row 9 it happens to be right
  (29,619) only because the output is one row per entity.
- **`_column_stats` corruption does not bite** (row 3 S6). Checked on every run
  over all 25 column names, not inherited.
- **`Table.infer_metadata` discards stypes set before it**, so `PINNED_STYPES`
  is applied after. Four columns needed pinning and two of them are load
  bearing: `PICKUP_ZONE_ID` and `DROPOFF_ZONE_ID` are 263/264-value integers
  over millions of rows, which is exactly the shape
  `kumo_core.stype_inference` calls *categorical* (under 4,000 distinct and
  under 5% unique). A categorical zone id is not a foreign key, so rows 12 and
  16 would lose their edge and row 16 its target. Row 2 hit the same rule from
  the other side with a 65,435-value port column.
- **`SnowflakeConnector(warehouse=...)` is inert locally**; the warehouse is set
  inside the session factory. It still participates in the connector id and so
  in the snapshot's graph key (row 1 S7).
- **`--no-publish-snapshots`** (`kumo.NoopArtifactLocation`) used for every run.
- **The in-tree SDK on `PYTHONPATH` is not optional.** Without it `kumoai_studio`
  resolves to the fat wheel and the run dies in the first second on
  `AttributeError: module 'kumoai_studio' has no attribute 'NoopArtifactLocation'`.
- **The static-column-target holdout is still required, and still 20%.** Rows
  1–4's rule transfers to row 9 unchanged: `PREDICT trips.TRIP_SECONDS FOR EACH trips.TRIP_ID` compiles its prediction table to
  `{"compare": {"alias": "col_1", "op": "is_null", ...}}` — verified from the
  compiled plan in `probe_nyc_query.py` before any data existed, and confirmed
  by the run, which returned exactly the 29,619 blanked rows and nothing else.
  Rows 12 and 16 need no holdout, as NOAA S2 predicted for temporal aggregate
  targets; row 16's prediction plan is `pass_through` with no filter at all.
  The holdout hash is salted (`trip_id || '/holdout'`) so it cannot correlate
  with the tier hash — row 2's first build nested two thresholds on one hash and
  produced a slice with zero labels.

### New to this block

- **No `timestamp` column is ever a feature** (S4). `kumo_core/pquery.py:810`
  excludes `stype in {ID, timestamp, unsupported}` from `feature_columns`, so
  the `Datetime()` encoder fifty lines above is unreachable on the portable
  path. Hour-of-day, day-of-week and calendar drift are all invisible to every
  model in this series; on NOAA and iot23 that cost little, on a taxi fixture it
  costs a great deal.
- **`RANK TOP 20` has no effect on the prediction output**, and the number it is
  ignored *in favour of* is 100, not the vocabulary size. NOAA S11 found
  multilabel returning the full class list; link prediction instead returns
  `min(100, |RHS|)` per entity, because `kumoai_studio/backend/local/prediction.py`
  calls `trainer.predict` with no `top_k` and
  `diskgraph_nested_trainer.py` defaults `pred_top_k` to 100. Measured: 26,400
  rows for 264 zones, exactly 100.0 per entity, when `TOP 20` was asked for.
  `BatchPredictionOptions.num_classes_to_return` exists in `kumoapi/jobs.py` and
  is read nowhere in `kumoai_studio/backend/local/`, so there is no way to set it.
- **The reported metric set is the task default regardless of the plan** (S8),
  for a reason unrelated to NOAA S7's boundary.
- **The training tables of every run against one graph snapshot live side by
  side** under `<snapshot>/train_tables/`, named by job id. Any tool that
  `rglob`s for `split_stats.json` and takes the first hit will report another
  run's split. This cost a `FORECAST 1` run its batch-prediction step here (the
  report crashed on the *previous* run's `null` validation stage), which is how
  it was noticed; `describe_split` now sorts by job id then mtime.
- **An interrupted ingest still wedges the graph permanently** (NOAA S10),
  reproduced here from an **OOM kill** rather than a deliberate interrupt, and
  it took out the next queued run as collateral (S9). Recovery is still
  `rm -rf <scratch>/diskgraph_snapshots/<graph key>` and nothing suggests it.
- **Snowflake `MEDIAN` over a 149-million-row held-out join is not free but is
  not the bottleneck either**: the whole-fixture row 9 baseline table
  (`probe_nyc_target.py --duration`) is one query over 745M rows and returns in
  under a minute on `KUMO_L`.

## Reproducing

```bash
source scripts/_env.sh  # resolves the monorepo SDK and root .venv

# One-time: the external stage, then one typed copy of all 745M trips and both
# dimension tables (284.4 s on KUMO_L).
"$KUMO_SDK_PYTHON" scripts/kaggle/nyc_fhv/probe_nyc_fixture.py
"$KUMO_SDK_PYTHON" scripts/kaggle/nyc_fhv/probe_nyc_raw.py        # unit + part shape
"$KUMO_SDK_PYTHON" scripts/kaggle/nyc_fhv/build_nyc_tables.py --raw-only

# What the three queries compile to, and which zone edge each traverses under
# both foreign-key orders. Costs a second and needs no data -- and it is what
# establishes S2 and S3 before anything is built.
"$KUMO_SDK_PYTHON" scripts/kaggle/nyc_fhv/probe_nyc_query.py --fkey-order both

# Every matrix claim, and what the three targets are worth with no model.
"$KUMO_SDK_PYTHON" scripts/kaggle/nyc_fhv/probe_nyc_stats.py --all
"$KUMO_SDK_PYTHON" scripts/kaggle/nyc_fhv/probe_nyc_target.py --all

# Build one tier and run one row on it, sampling free disk throughout.
# Row 9 tiers are hash samples over the whole span and carry a 20% holdout;
# rows 12 and 16 tiers are contiguous windows and carry none.
scripts/kaggle/nyc_fhv/run_nyc_tier.sh 9  r9tiny
scripts/kaggle/nyc_fhv/run_nyc_tier.sh 9  r9smoke
scripts/kaggle/nyc_fhv/run_nyc_tier.sh 16 z30
scripts/kaggle/nyc_fhv/run_nyc_tier.sh 16 z365

# The leak contrast (S4): identical features, so the difference is AutoML noise.
TAG=_leak scripts/kaggle/nyc_fhv/run_nyc_tier.sh 9 r9tiny --with-dropoff-time

# Row 12's three shapes (S5). Only the last completes end to end, and it is not
# a forecast -- and it predicts a constant.
scripts/kaggle/nyc_fhv/run_nyc_tier.sh 12 z30                       # FORECAST 7
TAG=_f1 scripts/kaggle/nyc_fhv/run_nyc_tier.sh 12 z30 --forecast-length 1
TAG=_f0 scripts/kaggle/nyc_fhv/run_nyc_tier.sh 12 z30 --forecast-length 0

# Reproduce the row 16 rejection deliberately (S2): with the drop-off edge
# declared last, the published query is "Nothing to predict".
scripts/kaggle/nyc_fhv/run_nyc_tier.sh 16 z30 --anchor-edge DROPOFF_ZONE_ID

# Re-score a finished run without retraining. Batch predictions are durable.
"$KUMO_SDK_PYTHON" scripts/kaggle/nyc_fhv/run_nyc_local.py --row 16 --scale z30 \
    --rescore /mnt/localssd/kumo-nyc/artifacts/predictions/<job>/predictions.parquet
```

Recovery, if a run is interrupted during ingest (NOAA S10 — the binary never
writes a `_phase_complete_context` marker, so every resume fails):

```bash
rm -rf /mnt/localssd/kumo-nyc/scratch/diskgraph_snapshots/<graph key>
```

Snowflake objects (account `TLB22710`): stage
`HYU.KAGGLE_FIXTURES.NYC_FHV_20GB`; tables `HYU.NYC_FHV.TRIPS_RAW`
(745,287,023 rows), `ZONES_RAW` (264), `PROVIDERS_RAW` (4), plus
`TRIPS_<TIER>` per tier and `TRIPS_<TIER>_H20` / `TRIPS_<TIER>_H20_TRUTH` for
row 9.

## Scripts

| file                   | purpose                                                                                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `probe_nyc_fixture.py` | stage the S3 fixture; manifest and inferred schemas                                                                                          |
| `probe_nyc_raw.py`     | the timestamp unit and the part-file shape, off five part files, before the expensive pass                                                   |
| `probe_nyc_query.py`   | what the three queries compile to, under **both** foreign-key orders, before any data exists (S2, S3)                                        |
| `probe_nyc_stats.py`   | every matrix claim checked: keys, both edges, span, density, columns, hash uniformity                                                        |
| `probe_nyc_target.py`  | what the targets are worth with no model: the `trip_seconds` identity, the zone-pair baseline, demand persistence, destination-set stability |
| `build_nyc_tables.py`  | one typed raw copy, then two tier families — hash samples for row 9, time windows for rows 12 and 16                                         |
| `run_nyc_local.py`     | the end-to-end run for any of rows 9, 12, 16, with the anchor-edge assertion                                                                 |
| `run_nyc_tier.sh`      | build + run one row on one tier unattended, with disk sampling                                                                               |
