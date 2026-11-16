# `kaggle_iot23_flow_malicious_10gb` on the SDK's local backend

Row 2 of [`kaggle-fixture-matrix.md`](../../../kaggle-fixture-matrix.md), run
end to end in-process, the same way [row 1](../ethereum/FINDINGS.md) was.

**Result: the pipeline works.** Three tiers complete end to end. But the headline number should not be quoted without §3: a model that
knows nothing but which capture a flow came from already scores 0.926 AUROC, and
the graph hands it that for free.

## Runs

|                            | smoke               | small           | part            |
| -------------------------- | ------------------- | --------------- | --------------- |
| flows (labeled + held out) | 163,106             | 3,254,288       | **16,266,495**  |
| hosts                      | 160,187             | 3,184,500       | 15,742,423      |
| held out for scoring       | 32,371              | 650,065         | 3,252,747       |
| malicious (labeled)        | 90.50%              | 90.51%          | 90.50%          |
| train                      | 132.9 s             | 1,801.2 s       | 4,454.5 s †     |
| predict                    | 2.5 s               | 23.6 s          | 198.4 s         |
| test AUROC / refit         | 0.8027 / **0.4807** | 0.9535 / 0.8654 | 0.7957 / 0.7474 |
| test AP                    | 0.9999              | 1.0000          | 0.9999          |
| **held-out AUROC**         | 0.9606              | **0.9996**      | **0.9860**      |
| peak disk                  | 0.21 GiB            | 1.77 GiB        | 7.52 GiB        |

† **Contended.** The `part` tier shared the 4 vCPUs for its whole duration with a
[row 3](../bccc/FINDINGS.md) run, so 4,454.5 s is an upper bound rather than a
clean measurement.

Held-out AUROC is measured against labels blanked out of the graph entirely
(§1), computed with the rank identity rather than `sklearn` (row 1 §2).

## 1. The static-target holdout is needed again

Exactly as [row 1 §1](../ethereum/FINDINGS.md). `PREDICT network_flows.is_malicious` is a static column target, so Kumo compiles the
prediction table to `WHERE is_malicious IS NULL`; the fixture labels all
325,309,946 flows, so batch prediction as published selects nothing.

`--holdout-pct 20` blanks a deterministic slice and keeps the true labels in a
`*_TRUTH` side table. **Second confirmed instance** — this is a property of
every static-column row in the matrix, not a quirk of one fixture. Rows 3, 4 and
10 will need the same.

One difference from row 1: `*_TRUTH` here holds only the held-out
`(flow_id, is_malicious)` pairs, not a full copy of the table. At these row
counts a full copy is another ~10 GB for two columns' worth of use.

## 2. The matrix row count is the sum of both tables

The matrix says 583,341,095 rows. Measured:

| table           |            rows |
| --------------- | --------------: |
| `network_flows` |     325,309,946 |
| `hosts`         |     258,031,149 |
| **sum**         | **583,341,095** |

So `network_flows` is 325M, not 583M. Same convention as row 1 (383,756,326 was
the sum across three tables), so it is consistent rather than wrong — but it
matters here, because §5 shows the entity count is what bounds the run, and the
entity table is `network_flows`.

Two matrix claims that *do* hold, unlike row 1's: `flow_id` is genuinely unique
(325,309,946 distinct, 0 null) and so is `host_id` (258,031,149 distinct, 0
null). No dedup needed.

## 3. The graph reconstructs the label channel that was dropped

**This is the finding that matters.** Held-out AUROC of 0.9996 on a
90.5%-positive target is high enough to require an explanation before it is
quoted as detection quality.

Three columns are dropped from the graph as label leaks: `label` (the string
form of the target — `'Malicious'` iff `is_malicious`, no exceptions in 325M
rows), `detailed_label` (the attack family, `'-'` exactly when benign, and the
target of matrix row 4), and `scenario_id` (names the IoT-23 capture).

`scenario_id` is dropped because a capture is one malware sample run against one
device, so it very nearly determines the label — per-scenario malicious rate is
above 0.99 or below 0.01 for most of the 23 scenarios.

**But dropping the column does not remove the information.** Measured on the
`small` tier:

|                                                                         |                                     |
| ----------------------------------------------------------------------- | ----------------------------------: |
| hosts appearing in exactly one scenario                                 | **3,180,323 of 3,184,500 (99.87%)** |
| mean scenarios per host                                                 |                              1.0013 |
| AUROC of "score = scenario's malicious rate, fitted on train rows only" |                          **0.9256** |

Both `source_ip` and `destination_ip` are edges into `hosts`, and a host almost
never appears in more than one capture. So the capture is recoverable from graph
structure alone, and recovering it is worth 0.926 AUROC with no model at all.

The model does add something over that baseline. But the honest description of
this task as constructed is *substantially "identify the capture"*, and the
graph is what makes that easy. Anyone quoting these numbers as IoT intrusion
detection should quote the 0.9256 structural baseline beside them.

`probe_iot23_leakage.py` measures all three rows above. It can do so because
`flow_id` is literally `'<scenario_id>:<uid>'` — the dropped column survives
inside the primary key. Keys are ID-typed and not featurized, so this is not
itself a leak, but it is worth knowing, and it is what makes the baseline
measurable.

**This applies to matrix rows 3 and 4 too.** Row 4 is the same graph with
`detailed_label` as the target; row 3 (BCCC) has the same `src_ip`/`dst_ip` →
`hosts` shape and should be checked the same way.

## 4. At a 90.5% base rate, average precision is not informative

The target is 90.5% *positive* — a rare-negative problem, the inverse of row 1's
1.78%-positive one. A majority-class classifier is already 90.5% accurate.

Average precision degenerates accordingly. At smoke scale the run reports
`test_ap=0.9999` alongside `test_auroc=0.8027`, and `test_refit_ap=0.9998`
alongside `test_refit_auroc=0.4807` — a **refit model that is worse than chance
while its AP still reads 0.9998**. AP is dominated by the base rate here; AUROC
is the only one of the two worth reading, and the refit instability at smoke
scale is real and should not be papered over.

It resolves with scale (refit AUROC 0.8654 at `small`), which is consistent with
the smoke tier simply being too small — 130,735 labeled rows with 12,422
negatives.

## 5. The entity is the fact table, so cost scales with rows

Row 1's entity table was fixed at 3.07M addresses no matter how many
transactions were added, so AutoML cost barely moved between tiers. Here the PQL
is `FOR EACH network_flows.flow_id` — the entity *is* the 325M-row fact table,
so every extra row is another entity.

Three measured points:

| tier  |      flows |       train | ×rows vs prev | ×time vs prev | peak disk |
| ----- | ---------: | ----------: | ------------: | ------------: | --------: |
| smoke |    163,106 |     132.9 s |             — |             — |  0.21 GiB |
| small |  3,254,288 |   1,801.2 s |         20.0× |         13.5× |  1.77 GiB |
| part  | 16,266,495 | 4,454.5 s † |          5.0× |          2.5× |  7.52 GiB |

**Scaling is sublinear, and it gets better with size.** The smoke→small step cost
13.5× for 20× the rows (an exponent of 0.87); small→part cost 2.5× for 5× the
rows (0.56). Fixed per-run overheads dominate the small tiers.

That makes the earlier estimate wrong. Extrapolating linearly from the first two
points gave "~50 hours at full scale", and that figure is **retracted**: carrying
the observed 0.56 exponent forward from `part` (20× more rows to reach 325M)
gives roughly **7 hours**, and even assuming scaling degrades back to linear it
is ~25 hours. The honest statement is that full scale is **plausibly reachable
in a day**, not the flat "not attempted" the two-point fit implied — but three
points with inconsistent exponents cannot pin it down, and the only way to know
is to run it.

Disk, extrapolated from `part` rather than `small`: 7.52 GiB for 16.27M rows is
0.46 GiB per million, so 325M rows is **~150 GiB** against 217 GiB free. Feasible
but without much headroom for the ingest transient.

† Contended for its whole duration, so the `part` time is an upper bound — which
means the sublinear trend is, if anything, understated.

## 6. Part files are one capture each

Each of the 23 part files holds exactly one `scenario_id` with its own
contiguous time range (`SCENARIOS=1` for all 23). This is the **opposite** of
row 1, where every part spanned the full history and one part was a
representative sample.

The practical consequence bit immediately: a slice built from a single part is a
single capture, and most captures are ~100% or ~0% malicious, so the first smoke
build produced a slice with no labeled rows left after the holdout and crashed
computing class balance. Slices must be sampled across scenarios — `SCALES` is
therefore expressed as parts-per-10,000 of `flow_id`, which samples the whole
fixture uniformly, not a prefix of it.

Sampling flows uniformly does thin each host's neighbourhood, which matters for
a graph model. It is survivable only because the degree distribution is extreme
— the scanning hosts have millions of flows each — so even a 1% sample leaves
them with thousands. Quote slice numbers against full-scale ones with that in
mind.

## 7. Smaller notes

- `event_time` is **microseconds** since epoch (`ts` is the same instant in
  float seconds, which is how the unit was settled). Reading it as nanoseconds
  silently yields the year 48,355,176 and then fails in the arrow decoder rather
  than at the cast.
- Data spans 2018-05-09 to 2019-09-21, 0 null timestamps.
- `uid` is Zeek's per-connection id, unique per row — dropped as noise with a
  high-cardinality encoder attached.
- Several columns infer to a surprising stype and are pinned by hand:
  `SOURCE_IP` (categorical → ID), `DESTINATION_IP` (text → ID),
  `DESTINATION_PORT` and `SOURCE_BYTES` (categorical → numerical).
- `scikit-learn` is still absent, so every run logs `Optuna search analysis failed`. Known, non-fatal, fixed upstream in
  [kumo-ai/kumo#30507](https://github.com/kumo-ai/kumo/pull/30507).
- The probe scans were expensive: reading the external stage repeatedly cost
  600–1300 s per scan on `KUMO_L`. `NETWORK_FLOWS_RAW` materializes the fact
  table once; query that, not the stage.

## Reproducing

```bash
source scripts/_env.sh  # resolves the monorepo SDK and root .venv

# One-time: external stage over the S3 fixture, then the raw fact table.
"$KUMO_SDK_PYTHON" scripts/kaggle/iot23/probe_iot23_fixture.py

# Build one tier and run it, sampling free disk throughout.
# Scales: smoke | small | part | half | full (parts per 10,000 of flow_id).
scripts/kaggle/iot23/run_iot23_tier.sh small 100

# What the score actually measures.
"$KUMO_SDK_PYTHON" scripts/kaggle/iot23/probe_iot23_leakage.py --scale small
```

Snowflake objects (account `TLB22710`): stage
`HYU.KAGGLE_FIXTURES.IOT23_10GB`; tables `HYU.IOT23.NETWORK_FLOWS_RAW`
(325,309,946 rows, ~10 GB) plus `NETWORK_FLOWS_<TIER>`, `HOSTS_<TIER>` and
`NETWORK_FLOWS_<TIER>_TRUTH` per tier.

## Scripts

| file                     | purpose                                                        |
| ------------------------ | -------------------------------------------------------------- |
| `probe_iot23_fixture.py` | stage the S3 fixture; manifest and inferred schemas            |
| `probe_iot23_stats.py`   | key uniqueness, class balance, per-part shape, leak candidates |
| `probe_iot23_leakage.py` | what the score measures: the scenario-only baseline (§3)       |
| `build_iot23_tables.py`  | materialize Snowflake tables at a chosen tier                  |
| `run_iot23_local.py`     | the end-to-end run                                             |
| `run_iot23_tier.sh`      | build + run one tier unattended, with disk sampling            |
