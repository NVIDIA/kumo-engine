# End-to-end lifecycle tooling

Data, metrics and baselines for the end-to-end flows, which train a model, save
it, and run batch inference and serving in a fresh process. The flows are
written the way a user would write them; this directory holds what they're
measured against. It's development tooling, not part of the `kumo.pytorch` wheel.

The end-to-end tests run in two tiers. T2a is the per-commit end-to-end run at
size S, and T2b is the nightly run at size M (the sizes are in the dataset table
below). The names match the `--t2a` and `--t2b` pytest options and the `t2a` and
`t2b` test markers.

| Path | What it is |
|---|---|
| `datasets/hm.py` | Command-line entry point for `kumo.pytorch.datasets.retail`, the H&M-shaped generator with a planted signal |
| `metrics.py` | AUROC, shared by the flows and the baseline so both score the same way |
| `baselines/kumo_engine_hm.py` | The script that records the kumo-engine baseline; run by a developer, not CI |
| `baselines/kumo_engine_hm_plan.json` | The pinned kumo-engine `ModelPlan` (one trial; only the seed varies) |
| `baselines/hm-s.json`, `baselines/hm-m.json` | The recorded baselines at S and M |
| `flows/hm_gnn.py`, `flows/hm_pragma.py` | The two flows: a GNN, and a model in the style of the published PRAGMA paper, pre-trained and then fine-tuned with LoRA |
| `flows/common.py` | What both flows share: the schema, the anchors, the caller's row lookup, the training loop |
| `../tests/e2e/` | Tests that the generator is deterministic and its ceiling is right; the T2a and T2b checks (`test_lifecycle.py`) and what differs between the tiers (`tiers.py`) |

## The H&M-shaped dataset

```bash
python -m e2e.datasets.hm --scale S --seed 0 --out /tmp/hm-s
```

Tables shaped like RelBench `rel-hm`: `customers`, `articles` and
`transactions` (`t_dat` is a whole-second timestamp). Each article also has a
`listed_at` time, before the first day, so `articles` is a clocked table that
graph sampling can reach (DiskGraph samples only clocked tables). The task is "does the
customer buy anything in the next 7 days", one anchor per customer per weekly
cutoff, with train, val and test split by cutoff time. The same transactions,
ordered by `(t_dat, transaction_id)` per customer, are the event sequences for
the sequence models.

| Scale | Customers | Articles | Transactions | Anchors (train/val/test) | Generation |
|---|---|---|---|---|---|
| XS (tests) | 200 | 60 | ~8k | 800 / 400 / 800 | < 0.1 s |
| S | 2,000 | 500 | ~50k | 32k / 4k / 8k | < 1 s |
| M | 50,000 | 5,000 | ~1.2M | 400k / 100k / 200k | ~1 s |

The daily purchase probability is a known function of past-visible behavior
(recency, 30-day frequency, product-type affinity), static attributes and a
slowly drifting latent rate; the `retail` module's docstring states it. For every anchor
the generator records `true_prob`, the probability of a purchase in the next 7
days given everything known at the cutoff, and its AUROC, the **ceiling**, with
its sampling standard error (`ceiling_auroc_se`, Hanley and McNeil). Because
`true_prob` also knows the latent rate, no model that reads only past rows can
beat the ceiling except by sampling noise. A model more than a few standard
errors above it is reading future rows: at S the test ceiling is 0.859 with a
standard error of 0.008, and a feature that counts sessions through the label
window scores 0.94. Leaks of two or three days stay under that band; the flows'
future-row check catches those. `tests/e2e/` recomputes `true_prob` row by row
from the written tables and the latent parameters, independently of the
generator, and checks that labels are calibrated against it.

The generator also emits ten **point lookups** in `metadata.json`: five very
active customers (`true_prob` above 0.9, expected high) and five dormant ones
that buy often until day 60 and never after (`true_prob` below 0.05, expected
low). The thresholds apply to `true_prob`, not to a model's score: kumo-engine
scores the active ones 0.82–0.95 and the dormant ones 0.06–0.18 against a test
mean of 0.12, so a model assertion should be relative (every active above every
dormant, active above the mean) rather than an absolute "low".

| File | Contents |
|---|---|
| `tables/*.parquet` | The graph tables a flow ingests |
| `anchors.parquet` | `sample_id`, `customer_id`, `cutoff`, `split` |
| `oracle/labels.parquet` | `sample_id`, `label`, `true_prob`: for scoring only |
| `oracle/latent.parquet` | The per-customer latent parameters: for tests only |
| `metadata.json` | Sizes, cutoffs, ceilings, point lookups, signal parameters, table hashes |

## The kumo-engine baseline

kumo-engine trains its GNN on the same anchors, labels and splits (a custom
training table), so its test AUROC is directly comparable with a kumo-pytorch flow's.
Its pinned plan samples the GNN flow's neighborhood: each customer's 16 most
recent transactions (`customers.customer_id->transactions.customer_id`: 16), then
each transaction's article (`transactions.article_id->articles.article_id`: 1),
nothing else (`default`: 0). The script declares `listed_at` as the articles'
time column, as this package's schema does. "Most recent" is kumo-engine's default
temporal strategy (a sampler setting, not a plan field), the same rule as this package's
`recent`. The neighborhood is defined the same way; the anchors drawn per epoch
differ. What still differs:
kumo-engine encodes every article column and its own model and training budget
(`training_budget` in `hm-s.json`: 63 steps of 512 per epoch, 6–10 epochs, against
the flow's full 32,000 anchors per epoch, 7–12 epochs).
It needs its own environment: kumo-engine pins `torch<2.13` and builds DiskGraph
from its own checkout. On macOS (no `faiss-gpu` wheel), with `KE` an empty
directory, `KUMO_ENGINE` a kumo-engine checkout and `SHA` the kumo-engine commit:

```bash
git -C $KUMO_ENGINE archive $SHA packages 3rdparty pyproject.toml | tar -x -C $KE
uv venv -p 3.12 $KE/.venv
uv build --wheel -o $KE/wheels $KE/packages/diskgraph   # needs Rust; on macOS also SDKROOT
uv pip install --no-sources -p $KE/.venv/bin/python 'torch<2.13' \
  $KE/wheels/diskgraph-*.whl $KE/3rdparty/kumo-wheels/torch_geometric-2.9.0-py3-none-any.whl \
  $KE/packages/kumo-api $KE/packages/kumo-pql "$KE/packages/kumo-core[faiss-cpu]"
uv pip install --no-sources --no-deps -p $KE/.venv/bin/python $KE/packages/kumo-engine
uv pip install -p $KE/.venv/bin/python pandas filelock typing_extensions pydantic
```

Then, from this package's root directory:

```bash
$KE/.venv/bin/python -m e2e.baselines.kumo_engine_hm \
  --kumo-engine-repo $KUMO_ENGINE --work /tmp/hm-baseline
```

It first checks that kumo-engine's own labels (from its predictive query) equal
the generator's at all 44,000 anchors. It then runs on CPU, trains once per seed (0, 1, 2) plus a
rerun of seed 0, and writes `baselines/hm-s.json`: kumo-engine's metrics, our
AUROC on the test anchors, the test prediction distribution, the point-lookup
scores, the resolved model plan, the training budget, the kumo-engine commit,
run times and hardware. `--gnn-runs DIR...` adds the GNN flow's epochs from
those runs' `train.json` to the budget. Fields left as `"???"` in the pinned plan are resolved by kumo-engine;
the script refuses seeds that resolve to different plans, and records the
resolved one.

kumo-engine is not deterministic on CPU: rerunning seed 0 moved single scores by
up to 0.21 (changing the seed: 0.16), and the test AUROC by up to 0.0025. The summary therefore covers every run, the
rerun included. Re-record the baseline
once per kumo-engine pin, and whenever the generator's output changes (a test
compares the recorded table hashes with a fresh S generation); the T2a and T2b
tests read it and never recompute it.

| Recorded at S (kumo-engine `5321556a`, CPU, Apple M-series) | Value |
|---|---|
| Test AUROC, 4 runs | 0.8250–0.8271 (mean 0.8259, std 0.0010) |
| Ceiling AUROC (test) | 0.8590 (standard error 0.0076) |
| Test prediction mean | 0.122–0.125 (positive rate 0.125) |
| Fit / predict 8k anchors | 10–19 s / 1.5–1.7 s per run |

## The flows (T2a)

Each flow is written the way a user would write it, over the S dataset on
DiskGraph, and runs every stage in its own process:

```bash
python -m e2e.datasets.hm --scale S --seed 0 --out /tmp/hm-s
python -m e2e.flows.hm_gnn --data /tmp/hm-s --work /tmp/hm-gnn
python -m e2e.flows.hm_pragma --data /tmp/hm-s --work /tmp/hm-pragma
```

| Flow | Contract | Model | Stages |
|---|---|---|---|
| `hm_gnn` | customer columns (attributes, recency, sessions over 7/30/90/365 days) and a two-hop neighborhood: the 16 most recent purchases, then each purchase's article (product type, index group, colour, base price); label: any purchase in the next 7 days | PyG `HeteroConv` with two `SAGEConv` layers over customer–purchase and purchase–article edges, so article information reaches the customer | train → infer → serve |
| `hm_pragma` | profile and the last 48 purchases as field-name/value tokens in one ID space, with calendar and elapsed-time inputs aligned to the events; masked values for pre-training, the same label for fine-tuning | fields → event → history Transformers; LoRA on every Transformer layer plus a new head | pretrain → finetune → infer → serve |

Training materializes the anchor table with `fit_split="train"`, stops early on
val AUROC and writes the artifact and the weights. It also scores the test
anchors in that process. `infer` loads both in a fresh process and scores the
test anchors in inference mode. `serve` builds one `ServingAssembler`, prints its
`rows_spec`, and assembles single requests from the rows the flow looks up by
following that spec. Fine-tuning starts from the
pre-trained artifact's bound contract, so every token ID is the pre-trained one.

The GNN's neighborhood has two hops, so T2a exercises multi-hop sampling end to
end, including the future-row and serving checks. The second hop works only
because the generator gives `articles` a clock: DiskGraph samples only clocked
tables and refuses `uniform` sampling, so a static dimension table such as
RelBench's real `articles` still cannot be a graph node on DiskGraph.

`pytest --t2a tests/e2e` runs both flows (three GNN seeds, two PRAGMA seeds) and
checks them; `--e2e-dir DIR` keeps the runs and `DIR/t2a/metrics.json`. On an Apple
M5 laptop it takes about 5 minutes: 23–33 s per GNN seed, 75–80 s per PRAGMA
seed, and the checks. Finished runs under `--e2e-dir` are reused, so use a
fresh directory after changing a flow.

| Check | What it asserts |
|---|---|
| Fresh process | the test batches' digest and every prediction equal the training process's |
| Serving | each of 16 requests, served from the rows the flow looks up by following `rows_spec`, gives exactly the reference (in-memory) batch-inference batch, built from every customer and article and the customer's purchases at any time |
| Batch inference | the flow's DiskGraph test predictions equal the model's on those reference batches within 1e-6 (summation order only); for the GNN, only the 9 of 16 requests whose 16 most recent purchases do not end inside a session (the point lookups are heavy buyers) |
| Future rows | adding purchases (of newly listed articles) just after, inside and beyond each test cutoff's label window changes no prediction: exactly for PRAGMA, within 1e-6 for untied GNN anchors; for every GNN anchor, the customer columns and sampled purchase times stay the same and the sampled articles are exactly the sampled purchases' own, all dated at or before the cutoff; untied anchors also keep the same articles |
| Late-listed article | every customer's newest purchase, one second before a test cutoff, is of an article listed after it: the purchase is sampled and the article is not, so the article hop filters on the article's own clock (on its own this hop has no other leak path here, since every generated article is listed before the data starts) |
| Refusals | weights refuse another contract's or another fit's state; a fitted state refuses another contract |
| Label | the contract's label equals the generator's at all 44,000 anchors |
| AUROC | both at least recency alone plus 0.005 and at most the ceiling plus 3 standard errors; seeds within 0.005. The GNN's difference to kumo-engine's lowest run on the same neighborhood is written to `metrics.json`, not gated: a band wide enough for kumo-engine's own run-to-run and per-pin variation sits below the recency floor |
| Predictions | every test anchor scored once, no NaN, not constant, mean within 1.5x of the positive rate, every active point lookup above every dormant one and above the mean |

The reasoning for each band sits next to its constant in
`tests/e2e/tiers.py`. Two known gaps are asserted as they stand,
so fixing one fails its test and the gap is lifted:

- Recent-sampling ties. The order among equal times isn't guaranteed
  (`docs/limitations.md`; on DiskGraph, excluded from
  G-INFERENCE-001). When a customer's 16th and 17th most recent purchases share
  a timestamp (one session), DiskGraph's choice follows its index, and rows added
  after the cutoff change it. The in-memory engine compares primary keys as
  strings. About one test
  anchor in six is affected; the GNN's batch-inference and future-row checks skip them,
  and for them the article hop follows whichever tied purchase was kept.
- Histories with no event. kumo-pytorch assembles a batch in which no sample has an
  event on both engines, but the flows' models read at least one event per batch, so the flows serve
  only customers who bought before.

## The nightly tier (T2b)

`pytest --t2b tests/e2e` runs the same flows and checks at M (50,000 customers,
1.25M purchases) with the M bands in `tests/e2e/tiers.py`. Each flow trains on a
fixed sample of a quarter of the train and val anchors (`sample_anchors` in
`datasets/hm.py`, seed 0: 100,000 and 25,000) with the same models and epochs as at S, and scores all 200,000 test anchors. The future-row
checks rebuild each of the four test cutoffs with the added purchases (50,000
anchors each); the label check covers all 700,000 anchors. Two seeds per flow.

Measured on an Apple M5 laptop: about 1.6 minutes per GNN seed (two hops),
5 minutes per PRAGMA seed (2 pre-train and 3 fine-tune epochs), and a few minutes of
checks over the finished runs. CI trains each flow and seed in its own job and
then checks them together; GitHub's runner has 2 vCPUs (AMD
EPYC 7763) and is 3x (GNN) to 6x (PRAGMA) slower, so the whole T2b workflow takes
about 35 minutes.

| At M, test anchors | Test AUROC |
|---|---|
| GNN (two hops), seeds 0 and 1 | 0.8340, 0.8337 |
| PRAGMA, seeds 0 and 1 | 0.8353, 0.8349 |
| kumo-engine on the same neighborhood (`baselines/hm-m.json`), 4 runs | 0.8316–0.8322 |
| recency alone | 0.8103 |
| ceiling | 0.8576 (standard error 0.0015) |

The kumo-engine baseline at M is recorded the same way as at S
(`--scale M`, with the same aligned plan; 63–77 s fit per run on the same laptop).
It trains on all 400,000 train anchors, but its pinned plan caps an epoch at 64
batches of 512 (10 epochs ran), so it sees fewer examples than the kumo-pytorch GNN's
125,000-anchor sample over its epochs. kumo-engine's own labels
equal the generator's at all 700,000 anchors.
