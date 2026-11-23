# End-to-end lifecycle tooling

Data, metrics and baselines for the end-to-end flows, which train a model, save
it, and run batch inference and serving in a fresh process. The flows are
written the way a user would write them; this directory holds what they're
measured against. It's development tooling, not part of the `kumo.pytorch` wheel.

Pull requests and nightly runs use the same [package suite](../../../../.github/workflows/kumo-pytorch-tests.yml)
to run package and distribution checks, H&M at sizes S and M, conformance shards
and six GPU throughput measurements on pinned real-data extracts. Both publish the same reports. To replay the nightly suite, manually dispatch this
workflow without inputs on main or the current head of an open same-repository PR.

For an on-demand CUDA, AML, full-binding or Streaming 25B run, select the
[manual benchmark workflow](../../../../.github/workflows/kumo-pytorch-benchmarks.yml)
and choose its mode. GitHub enables manual dispatch once that workflow exists
on the default branch.

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

## Pinned suite datasets

The suite stages the non-25B correctness pins before pytest. The private sources require a trusted repository run. Fork PRs fail the suite preflight before private staging.
For a new dataset, add a manifest with its S3 keys, sizes and SHA-256 hashes. The nightly role needs read access to those keys.
To fill the local cache, run these commands from the package root with the AWS CLI and read access. A complete cache works offline.

```bash
export KUMO_PYTORCH_DATA_ROOT=/tmp/kumo-pytorch-data
python -m e2e.datasets.staging e2e/datasets/pins/*.json --root "$KUMO_PYTORCH_DATA_ROOT"
python -m pytest -q tests/e2e/test_{aml,synthetic,mbd,streaming}.py
```

Benchmarks pass the files from `e2e.datasets.staging.verify(root, manifest)` to
`prepare_hi_small` (E01–E05, selected by `binding`) or `prepare_l01` (LI Medium) in `e2e.datasets.aml`. Both take
`(files, fresh_directory, anchor_rows=rows)` and return the contract, dataset and anchors.
For HI Small, fit on `train` with `encoder_cutoff` one second before the latest training prediction time.
For Synthetic Medium, `load_synthetic(root, entity_type)` and `prepare_synthetic(tables, fresh_directory, entity_type)`
in `e2e.datasets.synthetic` accept `"customer"` (S02, four weekly anchors), `"account"` (S01, three),
or `"ltv"` (S04, four LTV anchors). S04 adds one transfer shard and the v2 LTV anchor pair.
Its graph runs one second before prediction. All three share the original source shards. The loader checks
their event IDs against a full scan of all 1,610 published shards before applying a time cutoff.
To repeat the scan, pass `read_synthetic(files, entity_type)` the verified files with all shards keyed by
`<table>/<filename>`. Fit with `fit_split="train"` and `encoder_cutoff` one second before the latest training anchor.

For MBD, use `load_mbd(root, manifest)` and `prepare_mbd(tables, fresh_directory)`
from `e2e.datasets.mbd`. Use `M04_MANIFEST` for 30-day counts or `M03_MANIFEST` for category forecasts.
Both use transaction parts 00171 (fold 0), 00014 (fold 3), and 00196 (fold 4). M04 stages nine objects
totaling 1,029,139,562 bytes. M03 adds only the v1 anchor pair (963,739,896 bytes) to the shared nightly cache.
To verify the extract against all clients and transactions, add the original shards to the file mapping
using their paths from `extraction.json` and call `read_mbd(files, manifest)` again.

Streaming 25B stages two private Parquet files (3,965 bytes). `load_streaming(root)` reads viewers 3 and 873
and 36 sessions for the three forecast weeks. [`streaming-25b.json`](datasets/pins/streaming-25b.json)
records the original shard keys, sizes and hashes. The full-cohort digest covers 10,000 rows sorted by viewer ID,
each encoded as four little-endian int64 values (viewer ID, training count, validation count, test count).

For the opt-in full-source check, choose a cache with room for the 512,202,229,780 bytes of
source Parquet and a scratch directory with room to build the native index. Run from the package root:

```bash
KUMO_PYTORCH_25B_ROOT=/large/cache TMPDIR=/large/scratch \
  python -m pytest -q -o tmp_path_retention_policy=failed \
  'tests/e2e/test_streaming.py::test_streaming_weekly_session_forecast[25b-full]'
```

The existing stager downloads missing objects and verifies all 744 shards against the pinned source
manifest. The test checks the 30,000 weekly counts, compares indexed row counts for all seven tables
with the manifest, and replays viewers 3 and 873 over the three weeks.

The manual benchmark workflow offers five Streaming 25B modes on `main`
or a trusted `pull-request/<number>` branch.

| Mode | What runs |
| --- | --- |
| `smoke` | The four-viewer fixture, the two-viewer private extract and the full-source replay above. |
| `training` | Two same-seed 100-update CPU GraphSAGE runs on the first 32 training viewers, followed by checkpoint evaluation on validation and test. |
| `lifecycle` | A fresh process reloads fitted state and weights, checks target-free predictions, fine-tunes with frozen inputs and resumes a partially consumed training epoch. |
| `full` | Native delivery of all 30,000 anchors in source order, with their original identities, cutoffs, count targets and masks. |
| `all` | All four cases in one pytest invocation, sharing the verified source and native index. |

The job checks for 3 TiB free before setup. The cold smoke run used 2.83 TiB. To use a larger worker
or volume than the existing RTX runner and its temporary directory, set the repository variables
`KUMO_PYTORCH_25B_RUNNER` (Linux x64 runner label) and `KUMO_PYTORCH_25B_SCRATCH` (writable
directory) as needed. The worker uses `kumo-pytorch-25b/cache` under the scratch
directory. Populate it in advance or grant the worker read access to the pinned source objects.
JUnit records each case's duration and the shared staging, cohort and index times. For local runs,
select `tests/e2e/test_streaming.py::test_streaming_campaign[MODE]` for any of the three new cases.
Training and lifecycle use the package's PyG examples dependency.

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
uv pip install --no-sources --no-deps -p $KE/.venv/bin/python \
  $KE/packages/kumo-settings $KE/packages/kumo-engine
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

## The medium H&M tier (T2b)

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

## Suite training and results

The GPU matrix runs S02 (Synthetic nuFormer), L01 (AML LI Medium
Combined), S01 (Synthetic PRAGMA), M04 (MBD GraphSAGE), S04 (Synthetic
Combined) and M03 (MBD nuFormer). The package runner prepares DiskGraph and
loads batches from Parquet anchors on CPU. Each suite job stages the prepared
Parquet files named in `datasets/pins/throughput/` and fits on 8,192 training
anchors. With a batch size of 32 it warms up for 10 optimizer updates, times
the next 100 and evaluates 256 validation and 256 test anchors. Batches without
eligible targets count toward throughput and skip the optimizer update. For each anchor, the training
runner selects the latest 64 history events across its event sources and
delivers them oldest first. Each graph hop samples up to 64 recent neighbors
per relation and keeps smaller configured fanouts. These limits apply to model
inputs for training, validation and test. All 12 L01 count features use their
original time windows. The package correctness tests use the original histories.
HI Small retains its source and behavior checks. The 25B fixtures and campaigns
run on manual dispatch.

To reproduce an extract and its object hashes from staged original Parquet files,
run `python -m e2e.datasets.throughput_sources S04 --source-root /scratch/source --output /scratch/new-s04`
from the package directory. The command writes a local `pin.json` alongside the prepared tables.

Manual benchmark mode `binding-train` fits on the full registered training split.
It trains for two complete epochs, checks every sample ID, validates after each
epoch and tests after epoch two. Select one binding or use `binding=all` to run all six.
The manual `cuda` job also runs L01 training row 2,172,838 through this model
context and checks its 12 counts against the pinned transaction source.
The existing `KUMO_PYTORCH_BINDING_RUNNER` and `KUMO_PYTORCH_BINDING_SCRATCH`
variables select the Linux CUDA worker and writable volume. From a worker with
the full source staged in `KUMO_PYTORCH_DATA_ROOT`, the command is

```sh
python -m tests.e2e.runs binding-train S04 --e2e-dir /scratch/work --output report.json
```

The combined `results.md` and `results.json` contain per-head eligible counts
and mean losses for each window. The suite reports extract staging, verification,
index preparation and materialization/fit separately. Its throughput covers the
100 timed updates, including CPU batch loading and GPU synchronization. The
manual report includes staging and projection in ingestion and times both epochs.
Tokens count valid input history tokens (zero for graph-only M04). RAM is peak
process RSS, GPU utilization is the mean of device-wide NVML samples during
training, and SSD usage is allocated space for the staged binding and its
live projected tables and index.

## Package CI results and required check

The package suite publishes an attempt-specific
`kumo-pytorch-results` artifact. Its JSON and job summary identify the tested
revision, jobs, dataset selections and missing evidence. A pull
request can test a merge commit. The report records its source head separately.
Use **Re-run all jobs** when an entire new attempt needs acceptance.

After merge, download `results.json` from a completed scheduled run on
`main`. From the package directory, run the following with authenticated
read access through `gh`.

```sh
GITHUB_REPOSITORY=NVIDIA/kumo-engine python3 -m tests.e2e.ci_plan scheduled --results results.json
```

This verifies the actual workflow, revision, attempt and job conclusions. A
scheduled run that fails its tests still records that the schedule fired.

After verifying the PR and scheduled reports, a repository
administrator can add a branch ruleset targeting `main` with **Require status
checks to pass**. Select the exact check `kumo-pytorch-gate` from GitHub Actions,
keep the existing rulesets in force, and enable the new rule when ready.
The additional required-status-check entry is
`{"context":"kumo-pytorch-gate","integration_id":15368}`.
