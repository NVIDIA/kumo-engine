# `kaggle_bccc_source_destination_static_lp_20gb` on the SDK's local backend

Row 14 of [`kaggle-fixture-matrix.md`](../../../kaggle-fixture-matrix.md), run
end to end in-process the same way rows 1–5, 8, 9, 11, 12 and 16 were. The first
**static** link-prediction row in the series. It reuses [row 3](../bccc/FINDINGS.md)'s
fixture and row 3's fact table; **nothing here re-read the 20 GB stage.**

```
PREDICT  LIST_DISTINCT(network_flows.dst_ip)
RANK TOP 100
FOR EACH hosts.host_id
```

Verdict: **end to end at four tiers**, and the model is **statistically
indistinguishable from a global popularity ranking** and **2–3× below a
one-query co-occurrence baseline**. Unlike [row 16](../nyc_fhv/FINDINGS.md) the
task itself is *not* degenerate — it is the hardest and best-posed ranking
problem in the series — which makes the model's failure to clear a `GROUP BY` a
result about the model rather than about the fixture.

| headline                                                 | measured                                                                                                                            |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **as published the query does not compile at all**       | rejected because the published graph has a time column, not because of anything in the PQL (§2)                                     |
| **the anchor edge is again list order**                  | `SRC_IP` must be declared **last**; with `DST_IP` last the query is rejected as a loop (§3)                                         |
| **the task is not saturated**                            | median source host reaches **1** destination out of **407,244** candidate nodes; NYC row 16's median entity reached 215 of 264 (§4) |
| **model vs baselines**                                   | filtered MAP@100 **0.0400** at `lp_part` against popularity **0.0242** and co-source **0.1141** (§7)                                |
| **the noise floor is bigger than the popularity margin** | two runs on byte-identical data gave MAP@100 0.0299 and 0.0186 — a **38% swing** (§7)                                               |

## Runs

4 vCPU, 15 GB RAM, one L4, `/mnt/localssd` empty at the start, nothing else
running — timings are measurements, not upper bounds. A tier is a hash sample of
**source hosts** carried with all their flows (§6).

|                                  |       `lp_tiny` | `lp_tiny` (2nd) |        `lp_smoke` |             `lp_small` |                `lp_part` |
| -------------------------------- | --------------: | --------------: | ----------------: | ---------------------: | -----------------------: |
| source hosts                     |             978 |             978 |             3,927 |                 19,362 |               **78,167** |
| flows in the graph               |          27,946 |          27,946 |           302,425 |              1,548,875 |            **6,819,417** |
| candidate nodes (`hosts`)        |           1,429 |           1,429 |             8,570 |                 30,804 |               **98,166** |
| held-out pairs / scoreable hosts |     1,300 / 404 |            same |     4,798 / 1,697 |         28,704 / 8,404 |         124,623 / 33,656 |
| end to end                       |         **yes** |         **yes** |           **yes** |                **yes** |                **yes** † |
| table → DiskGraph index          |            11 s |            11 s |              17 s |                   61 s |                **432 s** |
| index on disk                    |         18.9 MB |               — |          271.6 MB |                1.41 GB |              **5.95 GB** |
| train                            |     **165.4 s** |         118.6 s |       **181.7 s** |            **214.0 s** |              **781.3 s** |
| predict                          |           1.3 s |           1.3 s |             2.0 s |                  3.0 s |                    6.9 s |
| train / val / test rows          | 848 / 189 / 215 |            same | 6,878 / 844 / 848 | 22,458 / 4,173 / 4,173 | 66,912 / 16,766 / 16,716 |
| peak RSS                         |         1.93 GB |               — |           2.24 GB |                3.85 GB |              **6.58 GB** |
| reported `test_map@100`          |          0.0323 |          0.0215 |            0.0337 |                 0.0302 |                   0.0306 |
| **held-out MAP@100 (filtered)**  |      **0.0299** |      **0.0186** |        **0.0288** |             **0.0277** |               **0.0400** |
| *popularity MAP@100*             |        *0.0287* |        *0.0287* |          *0.0255* |               *0.0252* |                 *0.0242* |
| ***co-source MAP@100***          |    ***0.0501*** |    ***0.0501*** |      ***0.0906*** |           ***0.1012*** |             ***0.1141*** |

† `lp_part` trained and predicted cleanly and its model scores are real. It was
then **OOM-killed at 12.7 GB in the block's own scoring code**, not in the
backend — the baseline builder materialised a filtered copy of a 22,854-element
candidate ranking per host (770M objects). Fixed with a bounded top-*k* walk; the
rescore of the durable predictions completes at **2.97 GB** peak. All baseline
figures above are from the fixed scorer.

The second `lp_tiny` column is `--top-k 20`. Because `RANK TOP k` reaches nothing
(§9), it is **byte-identical work** — so the difference between the two columns
is pure AutoML variance.

Deliberate failures, both reproduced on the real graph:

|                         |                                                                                                                  |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--anchor-edge DST_IP`  | **rejected before ingest** — *"Target network_flows.DST_IP creates a loop by pointing to the entity table"* (§3) |
| `--time-column`         | **rejected before ingest** — *"Time range required for LIST_DISTINCT with time column"* (§2)                     |
| `lp_full` (48.1M flows) | **not attempted**: measured RSS extrapolates to ~23 GB on a 15 GB box (§8)                                       |

## 1. The fixture, checked against the matrix

Everything row 3 measured holds; three claims are new to this row and all three
check out.

| claim                                      | measured (all 48,120,094 rows)                                                                                                                                              |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 48,590,026 rows                            | **48,120,094** deduplicated `network_flows` + **407,244** `hosts`; the matrix figure is the sum of the **raw** 48,182,782 flows and 407,244 hosts, as in every previous row |
| `hosts` = 407,244                          | **exactly** `COUNT(*)` of `SELECT src_ip UNION SELECT dst_ip`. `hosts` is not an independently published node list; it is the union of the two endpoint columns             |
| edges `src_ip`, `dst_ip` → `hosts.host_id` | **0 nulls on either**, no orphans by construction                                                                                                                           |
| key `network_flows.flow_id`                | unique in `NETWORK_FLOWS_RAW` — row 3's dedup already applied                                                                                                               |
| span                                       | 2018-02-14 07:28 → 2018-03-02 19:39, **17 days**                                                                                                                            |

### The graph is extremely asymmetric, and that is the whole story

|                                                                        |                                                            measured |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------: |
| distinct `src_ip`                                                      |                                                         **390,140** |
| distinct `dst_ip`                                                      |                                                          **39,343** |
| distinct `(src_ip, dst_ip)` pairs                                      |                                                       **3,417,218** |
| destinations per source: p5 / p25 / **median** / p75 / p95 / p99 / max |                                 1 / 1 / **1** / 3 / 22 / 94 / 3,641 |
| source hosts with exactly one destination                              |                                                 **223,449 (57.3%)** |
| source hosts with **more than 100** destinations                       | **3,629 (0.93%)** — so `RANK TOP 100` binds on under 1% of entities |
| sources per destination: median / max                                  |  4 / 9,564; **10,384** destinations are reached by exactly one host |

**Traffic is only mildly concentrated.** The busiest destination is reached by
9,564 of 390,140 sources (0.28% of pairs); the global top 10 covers 2.6% of
pairs, the top 100 **21.5%**, the top 1,000 76.7%. So a popularity ranking is a
real but weak baseline — not the near-perfect one NYC row 16 had.

### The graph is nearly bipartite by address space — NEW

|                                             | measured over 3,417,218 pairs |
| ------------------------------------------- | ----------------------------: |
| pairs whose endpoints share a **/16**       |                        **86** |
| pairs whose endpoints share a **/24**       |                        **41** |
| pairs whose destination is in `172.31/16`   |         **2,416,480 (70.7%)** |
| distinct destination IPs inside `172.31/16` |                       **453** |
| pairs where the reverse pair also exists    |           844,626 (**24.7%**) |

The CSE-CIC-IDS2018 testbed is external attackers/clients → a fixed internal
victim range, and internal hosts → external services. Internal↔internal is 86
pairs in 3.4 million. **A subnet-prefix feature is therefore worthless here**,
which is unusual and worth stating: on a real enterprise capture it would be the
first thing to try.

Row 3's open leakage question transfers but weakens: a source host touches
**1.73** capture sessions on average and 68.8% touch exactly one (against
iot23's 99.4%), so "predict the destination" is not simply "predict the capture".

### Columns, and the stat-token workaround — now verified from the model

Row 3 §6's `_column_stats` defect bites this fixture hardest — 142 of 323 names
collide. The tier tables rename via row 3's `stat_safe_name` (`count→cnt`,
`median→mid`, `mean→avg`, `min→lwr`, `max→upr`, `std→sdv`), imported rather than
copied. **This block closes the loop by reading the statistics back out of the
trained model**, `published-model/model/encoders.json`:

```
PAYLOAD_BYTES_AVG    MaxLogNumerical {COUNT 27946, MEAN 36.068, STD 61.880, MIN 0.0, MAX 833.0}
PACKETS_CNT          MaxLogNumerical {COUNT 27946, MEAN 5.697,  STD 5.357,  MIN 1.0, MAX 72.0}
```

Four distinct, plausible statistics per column — under the fixture's own names
these would all have collapsed onto one token. **This is still a deviation from
the published column names.**

Encoded columns: `network_flows` **318**, `hosts` **0** (verified directly from
`encoders.json`). `hosts` is one column wide and is *both* the entity table and
the RHS candidate table, so every node in the ranking is featureless.

## 2. Row 14 as published does not compile — and the obstruction is the graph's time column — NEW

`probe_bccc_lp_query.py` compiles the published query against a synthesized
portable graph in about a second, with no data. Under the published graph —
`network_flows` carrying `event_time` — the query is **rejected**:

```
Time range required for LIST_DISTINCT with time column. When using LIST_DISTINCT
on a table with a time column (EVENT_TIME), you must specify a time range.
```

The full 2×2, measured (`--fkey-order both --no-time-column --variants`):

| `network_flows` time column | last-declared fkey | verdict                                                                                            |
| --------------------------- | ------------------ | -------------------------------------------------------------------------------------------------- |
| `EVENT_TIME`                | `SRC_IP`           | **rejected** — time range required                                                                 |
| `EVENT_TIME`                | `DST_IP`           | **rejected** — *two* errors: loop **and** time range                                               |
| *(none)*                    | `SRC_IP`           | **`static_link_prediction`**, `rhs_key = network_flows.SRC_IP`, `RandomLinkSplit([0.8, 0.1, 0.1])` |
| *(none)*                    | `DST_IP`           | **rejected** — creates a loop                                                                      |

So **"static link prediction" is a property of the graph, not of the query**, and
the same graph cannot express row 14 and the matrix's temporal-LP shape: adding a
window to the target on the time-less graph is rejected with *"Missing a 'Create
date' column"*. A user following the graph contract literally — build one BCCC
graph, run both rows against it — cannot.

Two neighbouring cells confirm NOAA §2's table from the side NOAA could not reach:

- `CLASSIFY TOP 100` instead of `RANK` is rejected in **every** graph variant:
  *"static usage of LIST_DISTINCT is only possible for RANK problems"* — NOAA hit
  this from the non-foreign-key side; row 14's target *is* a foreign key and
  still cannot use `CLASSIFY`.
- `FOR EACH network_flows.FLOW_ID` is rejected: *"There is no foreign key from
  table network_flows to table network_flows"*.

### NOAA's open `has_edges` question, answered

NOAA read a branch in `diskgraph/compiler.py` where a *static aggregate*
prediction table filters to entities with `has_edges = False`, and reported it as
a code reading because no NOAA query reached it. Row 14 is a static aggregate
target and **the branch does not fire**: the compiled plan is `filters=[null]`
and batch prediction returned **all** candidate nodes at every tier, including
destination-only hosts with no outgoing edge. Link prediction takes the
`lp_prediction_mode='LHS'` path instead. **Row 14 needs no holdout to produce
predictions** — but it needs one for them to mean anything (§6).

## 3. The anchor edge, decided by list order again — NEW confirmation, static form

NYC §2 established the mechanism on temporal LP: `to_api_graph` emits one
`ColumnKeyGroup` per foreign key, `join_validator.py::_find_keys` does
`keys = curr_keys` (assign, not extend), the last-declared fkey wins, and the
"Ambiguous link" check below is unreachable. There is no PQL syntax for it —
`PQLGrammar.g4` has no `VIA`/`USING`/`THROUGH`.

**It transfers to the static form exactly.** The matrix's instruction —
*"explicitly select the keyed `src_ip` edge as the `FOR EACH` anchor"* — is
implemented as:

```python
edges=[dict(src_table='network_flows', fkey='DST_IP', dst_table='hosts'),
       dict(src_table='network_flows', fkey='SRC_IP', dst_table='hosts')]   # anchor LAST
```

and every run asserts the outcome rather than trusting it:

```
network_flows fkeys (declaration order): ['DST_IP', 'SRC_IP']
resolved join key : network_flows.SRC_IP
task_type         : static_link_prediction (link_prediction)
rhs node type     : hosts
```

The graph is perfectly symmetric, which makes the demonstration unusually clean:
with `DST_IP` declared last, `PREDICT LIST_DISTINCT(network_flows.SRC_IP)`
compiles and `LIST_DISTINCT(DST_IP)` is rejected; with `SRC_IP` last it is the
other way round. **Nothing about the data decides it. The order of a Python list
does.**

Here the wrong order fails **loudly**, which is luck: NYC row 12 showed the same
mechanism failing *silently* on an aggregate target.

## 4. The saturation measurement — the task is *not* degenerate — NEW

NYC row 16 lost to a `GROUP BY` because its median entity reached 215 of 264
possible destinations. The same measurement here, before any training:

|                                              |     NYC row 16 |                               **BCCC row 14** |
| -------------------------------------------- | -------------: | --------------------------------------------: |
| candidate universe                           |      264 zones | **407,244 hosts** (39,343 ever a destination) |
| median truth-set size                        |        **215** |                                         **1** |
| median truth as a share of the candidate set |      **0.814** |                                 **0.0000025** |
| `RANK TOP k` asked for                       |             20 |                                           100 |
| recall@k ceiling by arithmetic               | 20/215 = 0.093 |                 **1.0** for the median entity |

**This is the most informative link-prediction instance in the series**: every
metric is unsaturated, MAP@1 and recall@100 both move, and there is real headroom
between the baselines. That is also why the numbers are all small. MAP@100 of
0.03–0.11 is not a broken run; it is what a genuine 1-in-100,000 ranking problem
looks like.

## 5. What `RandomLinkSplit` actually does, and what it does not — NEW

`kumo_core/pquery.py::_default_split` gives `static_link_prediction` a
`RandomLinkSplit([0.8, 0.1, 0.1])`. Three things measured:

**It really is a link split, not an entity split.** The same host appears in more
than one stage with disjoint `TARGET1` lists and a cumulative
`exclude_target_keys` — the standard filtered protocol. At `lp_part`, **11,069 of
16,716 test entities are also in train**. Excluded keys grow as expected: train 0,
val 333,286, test 375,118. So the reported `test_map@k` is a legitimate held-out
link metric, and it agrees with this block's independent holdout to within the
noise floor.

**The val and test edges are still in the DiskGraph index.** The index is built
from the flow table once; the split happens afterwards, in the training table.
Nothing removes an edge from message passing — the classic transductive LP leak.
The two independent measurements agreeing bounds how much it is worth here, which
appears to be little.

**The published resolved plan advertises the wrong split family.**
`resolved-model-plan.json` reads `training_table_generation.split = 'RandomSplit([0.8, 0.1, 0.1])'` — the `ModelPlan.from_default` value — while the
split actually used is `RandomLinkSplit`, because `_split_options` only reads the
plan's split when the *caller's* dict carries an explicit `split_family`. A user
reading the published artefact to find out how their model was validated is told
the wrong thing.

**Row 14 is the first BCCC row whose split is not pathological.** Row 3 §3's
capture-aligned temporal split put one class in validation and made `test_acc` a
constant predictor's score. A link split cannot do that. `stage_seed_times` are
all `[0, 0]` — a static query has no time at all.

## 6. Tiering and the holdout, both of which had to be designed for this row — NEW

**A tier is a hash sample of source hosts, not of flows.** Row 3's tiers hash
`flow_id`; that is right for a per-flow static column target and wrong here,
because the target is the set of destinations a host reaches and thinning its
flows changes its label. With a median degree of 1, a 1% flow sample would delete
most of the target outright. The predicate is
`MOD(ABS(HASH(src_ip)), 10000) < permille10`, so a host is wholly in or wholly
out and the tiers nest.

| tier       | permille10 | source hosts |      flows | distinct dst |
| ---------- | ---------: | -----------: | ---------: | -----------: |
| `lp_tiny`  |         25 |          978 |     34,845 |          451 |
| `lp_smoke` |        100 |        3,927 |    363,990 |        4,670 |
| `lp_small` |        500 |       19,362 |  1,931,972 |       11,857 |
| `lp_part`  |       2000 |       78,167 |  8,507,450 |       22,854 |
| `lp_full`  |      10000 |      390,140 | 48,120,094 |       39,343 |

**The holdout is over distinct `(src_ip, dst_ip)` pairs.** Row 14 needs none to
produce predictions (§2) but needs one for them to mean anything: a static target
is "every destination this host ever reaches", which is exactly what the graph
contains, so scoring against it measures memorisation.

Two invariants, and **the second was a correction made mid-block**:

- every eligible source host gives up at least one destination and keeps at least
  one — a per-entity leave-*n*-out split rather than a uniform 20% of pairs,
  because 57.3% of hosts have degree 1;
- **every destination keeps at least one source.** The first version constrained
  only the source side, and at `lp_smoke` **50.2% of held-out destinations then
  appeared nowhere else in the graph** — isolated, featureless nodes that no model
  can rank and no baseline can propose. Half the ground truth was unreachable by
  construction. With the destination-side constraint the figure is **0.0% at
  every tier**, which the build asserts and prints.

## 7. Results, and why the co-source baseline is the number that matters — NEW

Two protocols: **unfiltered** (the host's retained destinations are still in the
list and are free hits) and **filtered** (everything the host still reaches is
struck from every ranking first — the standard LP protocol, and **the honest
column**).

Two no-model baselines on the same rankings with the same metric code:
**popularity** (one global ranking by distinct source count — the
no-personalisation floor) and **co-source** (an item-item collaborative filter:
score a candidate by how often it co-occurs with the destinations this host
already reaches, backing off to popularity — one self-join, and exactly the
structure the GNN is being paid to learn).

**Filtered, at `lp_part`** (33,656 hosts, 124,623 held-out pairs):

|                  |   k |     prec@k |   recall@k |    **MAP@k** |   **NDCG@k** |
| ---------------- | --: | ---------: | ---------: | -----------: | -----------: |
| **row 14 model** |   1 |     0.0313 |     0.0099 |   **0.0313** |   **0.0313** |
|                  |  20 |     0.0217 |     0.1253 |   **0.0350** |   **0.0645** |
|                  | 100 |     0.0177 |     0.3972 |   **0.0400** |   **0.1208** |
| *popularity*     | 100 |     0.0117 |     0.2594 |     *0.0242* |     *0.0773* |
| ***co-source***  |   1 | **0.1120** | **0.0589** | ***0.1120*** | ***0.1120*** |
|                  | 100 |     0.0168 | **0.4707** | ***0.1141*** | ***0.2008*** |

**The model loses to the co-source ranking at every cut-off, by 2.9× at MAP@100
and 3.6× at MAP@1.**

MAP@100 across all four tiers, filtered:

| tier       | scored hosts |      model | popularity | **co-source** |
| ---------- | -----------: | ---------: | ---------: | ------------: |
| `lp_tiny`  |          404 |     0.0299 |     0.0287 |    **0.0501** |
| `lp_smoke` |        1,697 |     0.0288 |     0.0255 |    **0.0906** |
| `lp_small` |        8,404 |     0.0277 |     0.0252 |    **0.1012** |
| `lp_part`  |       33,656 | **0.0400** |     0.0242 |    **0.1141** |

**Scale helps the baseline and barely helps the model.** Co-source goes 0.0501 →
0.1141 (2.3×) as co-occurrence statistics thicken; the model goes 0.0299 → 0.0400
(1.3×), and is flat or falling across the first three tiers. The gap widens with
scale.

### The noise floor is larger than the model-vs-popularity margin

Because `RANK TOP k` reaches nothing (§9), the `--top-k 20` run at `lp_tiny` is
**byte-identical work** to the `--top-k 100` run. It is a pure re-draw of the
AutoML trajectory:

|                             |   run A | run B (`TOP 20`) | difference |
| --------------------------- | ------: | ---------------: | ---------: |
| reported `test_map@100`     |  0.0323 |           0.0215 |   **−33%** |
| held-out MAP@100 (filtered) |  0.0299 |           0.0186 |   **−38%** |
| training time               | 165.4 s |          118.6 s |       −28% |

**A 38% swing on identical data.** The model's margin over popularity is +4% at
`lp_tiny`, +13% at `lp_smoke`, +10% at `lp_small` and +65% at `lp_part`. Only the
`lp_part` margin is outside the noise band, and it rests on one run. The
defensible statement: **the model is indistinguishable from a global popularity
ranking at three of four tiers, may clear it at the fourth, and is clearly and
consistently below the co-source `GROUP BY` at all four.**

### Why the model underperforms, mechanically

`hosts` contributes **zero** encoded columns, so a candidate destination is a
bare lookup embedding. The only information distinguishing candidates is graph
position, and `num_neighbors = [12, 12]` fixed fanout on a graph whose median
host has one edge and whose busiest destination has 9,564. Meanwhile
`train_map@100 = 0.9680` against `test_map@100 = 0.0323` — the lookup embeddings
memorise the training links almost perfectly and generalise almost not at all.
The co-source baseline is exactly the 2-hop count the GNN is failing to aggregate.

## 8. Cost, and where the ceiling is

|                                   |                                                                                       measured |
| --------------------------------- | ---------------------------------------------------------------------------------------------: |
| tier build off row 3's fact table |                                                         **5–30 s** on `KUMO_L`, all four tiers |
| Snowflake table → DiskGraph index | 11 / 17 / 61 / **432 s** — ~63 µs per flow at scale, against NYC's 2.5 µs, because 321 columns |
| **DiskGraph index**               |                                                               **~865 bytes per flow** marginal |
| train                             |                              165 / 182 / 214 / **781 s** — ~10 ms per source host at `lp_part` |
| peak RSS                          |                                                               1.93 / 2.24 / 3.85 / **6.58 GB** |
| peak `/mnt/localssd`              |                                              0.15 / 0.55 / 2.18 / **8.88 GB** of 217 available |

**Disk is not the constraint.** The whole fixture would index into roughly
**33–42 GB** against 209 GB free.

**Memory is, exactly as NYC found.** Marginal RSS between `lp_small` and
`lp_part` is **0.52 GB per million flows**, putting the full fixture (~38.5M
flows after the holdout) at **~23 GB** on a 15 GB box. The ceiling is around
**21 million flows**, under half the fixture. Stated as a linear extrapolation
from two points, not as a measurement.

Time would not have been binding: 781 s for 78,167 hosts extrapolates to roughly
**65 minutes** of AutoML for all 390,140, plus ~40 minutes of ingest. That is the
opposite of row 3, where a 48M-row *fact*-entity run extrapolated to ~66 hours —
because row 14's entity table saturates at 407,244 while row 3's was the 48M-row
fact table.

## 9. Carried over, confirmed again on a fifth task type

- **`training_job.metrics` is still ignored** (NYC §8), and this row makes it
  unmistakable in the *other* direction. `resolved-model-plan.json` records
  `metrics: ['map@10','map@100','recall@100']` — the request, verbatim — and the
  run then reports **all twelve `static_link_prediction` defaults** plus their
  `refit_` twins: 9 metric names nobody asked for, 0 missing.
  `kumo_core/training/workflow.py::_resolved_task_options` reads the metric list
  from a `task` model-plan section that `kumoapi.model_plan.ModelPlan` does not
  have. **A check that only looks for missing names calls this "applied"**; the
  run script now reports both directions.
- **`tune_metric` does arrive** (`map@100`). The fix in the working tree holds.
- **`RANK TOP k` has no effect on the prediction output.** `--top-k 20` produced
  **exactly 100 rows per entity**, at every tier. Verified directly:
  `kumoai_studio/backend/local/prediction.py` calls `trainer.predict(output_dir=..., task=task)` with no `top_k`, and `BatchPredictionOptions.num_classes_to_return`
  exists in `kumoapi/jobs.py` and is **read nowhere** in `kumoai_studio/backend/local/`.
  NYC found this on temporal LP; unchanged on static LP.
- **`num_entities_predicted` reports rows, not entities** (row 3 §5): 9,816,600
  for 98,166 hosts × 100.
- **`validation` metrics come back empty** in the SDK's metric grouping — a sixth
  fixture, even though `val_map@100` is logged every epoch.
- **`scikit-learn` absent** → `Optuna search analysis failed` on every run.
  Non-fatal; the tier script waits on exit status, not on a `Traceback` grep.
- **`Table.infer_metadata` discards stypes set before it**, so pins go after. 60
  pinned, 22 moved. `SRC_IP`/`DST_IP` must be pinned to `ID`: left alone an IP
  string reads as free *text*, and here that would cost the edge **and** the
  target, since `static_link_prediction` is reached only when the `LIST_DISTINCT`
  target resolves to a foreign key.
- **`SnowflakeConnector(warehouse=...)` is inert locally.**
- **`--no-publish-snapshots`** used for every run; **the in-tree SDK on `PYTHONPATH`
  is not optional.**

### New to this block

- **A static aggregate target gets no prediction filter at all** — NOAA's
  `has_edges = False` branch is still unreached, and every candidate node is
  scored (§2).
- **`refit_trainval: True` is the `static_link_prediction` default**, so every run
  reports a second `refit_*` metric set. At `lp_tiny` the refit model was *worse*
  (0.0263 vs 0.0323); at `lp_part` better. Inside the noise floor either way.
- **`encoders.json` is `{"encoders": {table: {...}}, "target": ...}`**, not a bare
  table mapping. NYC's reader assumed the latter and would report "2 encoded
  columns" for a graph with 318. Verified directly.
- **The obvious way to write a filtered LP baseline is an OOM.** Materialising a
  filtered copy of the candidate ranking per entity is O(entities × candidates);
  at `lp_part` that is 770M objects and 12.7 GB. A bounded top-*k* walk does the
  same job at 2.97 GB. Worth stating because the *backend* survived `lp_part`
  comfortably at 6.58 GB — the analysis code was the thing that fell over.

## Reproducing

```bash
source scripts/_env.sh  # resolves the monorepo SDK and root .venv

# What row 14 compiles to under both foreign-key orders and with/without a time
# column. One second, no data -- and it is what establishes §2 and §3.
"$KUMO_SDK_PYTHON" scripts/kaggle/bccc_lp/probe_bccc_lp_query.py \
    --fkey-order both --no-time-column --variants

# Every matrix claim, and what the target is worth with no model (§1, §4).
"$KUMO_SDK_PYTHON" scripts/kaggle/bccc_lp/probe_bccc_lp_stats.py --all

# Build one tier and run row 14 on it, sampling free disk and RSS throughout.
# Scales: lp_tiny | lp_smoke | lp_small | lp_part | lp_full.
scripts/kaggle/bccc_lp/run_bccc_lp_tier.sh lp_tiny

# The two deliberate rejections (§2, §3), both before ingest.
"$KUMO_SDK_PYTHON" scripts/kaggle/bccc_lp/run_bccc_lp_local.py \
    --scale lp_tiny --anchor-edge DST_IP --no-publish-snapshots
"$KUMO_SDK_PYTHON" scripts/kaggle/bccc_lp/run_bccc_lp_local.py \
    --scale lp_tiny --time-column --no-publish-snapshots

# The noise floor (§7): TOP 20 is byte-identical work.
TAG=_top20 scripts/kaggle/bccc_lp/run_bccc_lp_tier.sh lp_tiny --top-k 20
```

Recovery, if a run is interrupted or OOM-killed during ingest (NOAA / NYC — no
`_phase_complete_context` marker is ever written, so every resume fails):

```bash
rm -rf /mnt/localssd/kumo-bccc-lp/scratch/diskgraph_snapshots/<graph key>
```

Snowflake objects (account `TLB22710`): row 3's `HYU.BCCC.NETWORK_FLOWS_RAW`
(48,120,094 rows, **not rebuilt**), plus `NETWORK_FLOWS_LP_<TIER>`,
`NETWORK_FLOWS_LP_<TIER>_TRUTH` and `HOSTS_LP_<TIER>` per tier.

## Scripts

| file                      | purpose                                                                                                                                  |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `probe_bccc_lp_query.py`  | what row 14 compiles to under both foreign-key orders and with/without a time column, before any data exists (§2, §3)                    |
| `probe_bccc_lp_stats.py`  | every matrix claim, the saturation measurement, popularity concentration, subnet and capture structure (§1, §4)                          |
| `build_bccc_lp_tables.py` | source-host tiers off row 3's fact table, with the two-sided pair holdout (§6)                                                           |
| `run_bccc_lp_local.py`    | the end-to-end run, with the anchor-edge assertion, split/encoder/plan readbacks, and filtered/unfiltered scoring against both baselines |
| `run_bccc_lp_tier.sh`     | build + run one tier unattended, sampling free disk and RSS                                                                              |
