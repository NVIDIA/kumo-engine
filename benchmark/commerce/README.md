# Commerce training and prediction

Run the commerce graph through ingestion, training-table generation, model fitting and batch prediction.
The reference measurement used the October 9, 2026 runtime build `20261009+9a0321c`, Python 3.12.3 and Torch 2.9.1.
Its completed run and separate prediction probe are summarized in `reference-results.json`.

## Contents and prerequisites

The downloadable folder places `build_graph.py` and `run_e2e.py` in `scripts/`.
In the repository these scripts are beside this README.
The runner uses the installed Kumo Engine runtime, its NumPy/PyArrow dependencies and Python's standard library; the included generator also uses SciPy.
No DuckDB, package installer, cloud CLI, notebooks or credentials are included.
Use the Engine environment setup guide to install a compatible runtime.
This benchmark harness instruments private trainer configuration to pin loader workers and collect timings; it is version-sensitive and is not a new supported public API.
A run on a newer engine revision is a new trial, not a rerun of the archived binary.

The full reference used one g6e.48xlarge with 192 vCPUs, 1.5 TiB nominal RAM, eight L40S GPUs and local NVMe scratch.
Provide the original local commerce Parquet directories and `_eval/predict_ids_5M.parquet`.
Use the included generator for representative data and evaluation IDs.
The historical runtime binary and measured data are not included.
For a smoke test use a smaller compatible rung and a short budget, recording it separately from the reference.

## Generate representative data

The `generator/` directory contains the original commerce generation algorithm with an invented placeholder profile.
It creates the same seven-table schema and workload family, but it does not recreate the original data distribution, row counts, bytes or benchmark timings.
No real-derived aggregate profile or original dataset is included.
NumPy, PyArrow and SciPy are already direct or transitive dependencies of Kumo Engine; no additional package install or version override is requested.
Use the compatible versions installed with your Kumo runtime and retain the versions recorded in each manifest.

Start with the tiny configuration, which reduces each market to 200 customers.
Run from the extracted folder using a new data destination.
The generator is invoked from its own directory so its `scalebench` module imports resolve:

```bash
cd generator
python -m scalebench.generate run --graph commerce --rung smoke --shards 4 --config smoke-config.json --root /path/to/generated-data --workers 1
python verify.py --root /path/to/generated-data --rung smoke
cd ..
python scripts/make_eval_ids.py --root /path/to/generated-data/commerce/smoke --out /path/to/generated-data/commerce/smoke/_eval --test-seeds 100 --score-customers 100 --workers 1
```

Use `/path/to/generated-data/commerce/smoke` as `DATA` for a smoke run.
For a larger representative dataset, run from `generator/` with the standard `config.json`, omit `--shards`, and choose `--rung 100M`, `1B` or `50B`.
Those are target row counts, rounded to whole market blocks, and the full 50B setting is a large workload.
Set `--workers` to fit your available memory; start with one.
Generation, evaluation-ID preparation and data verification are outside the benchmark timers.
The verifier checks SHA-256, schemas and row counts, not the original validator's statistical-distribution suite.
The original DuckDB-based validator is not included.
Synthetic `_ground_truth/` output is diagnostic hidden state and must never be used as model input.

## Run the downloaded folder

Set the paths to existing data and local SSD scratch, plus a new result directory.
The original graph definition and recommendation query are in `scripts/build_graph.py`.
The following command requests the large historical training budget on your generated data; it is not the tiny smoke test.
The task predicts the top ten merchants a customer will purchase from during the following seven days.

```bash
DATA=/path/to/commerce/50B
SCRATCH=/path/to/local-ssd/commerce-scratch
OUT=/path/to/new-commerce-results
python scripts/run_e2e.py --graph commerce --data "$DATA" --scratch "$SCRATCH" --out "$OUT" \
  --epochs 1 --max-steps 7000 --max-val-steps 700 --max-test-steps 700 \
  --batch-size 64 --hops 6 --fanout 24,8,6,8,6,8 --seed 1 --no-explain \
  --num-workers 8 --predict-workers 4 --predict-loader-workers 4 \
  --predict-ids "$DATA/_eval/predict_ids_5M.parquet"
```

This requests the recorded training budget and the four-scorer prediction configuration.
Training consumed 448,000 seed examples, not all training-table rows.
The prediction subset must contain the original five million IDs, and its generated anchor must match `2024-12-25 23:59:59`.
Check the recorded timestamps before comparing timings.
The archived prediction probe separately opened a saved snapshot and model; this combined invocation does not reproduce that probe's startup timer.

For ingestion only, append `--stages snapshot --num-workers 0` and use another output directory.
For a small workflow smoke test, use the generated `smoke` dataset, one GPU and a new output directory:

```bash
python scripts/run_e2e.py --graph commerce --data "$DATA" --scratch "$SCRATCH" --out "$OUT" \
  --epochs 1 --max-steps 2 --max-val-steps 1 --max-test-steps 1 \
  --batch-size 8 --hops 2 --fanout 4,2 --num-workers 1 --predict-workers 1 \
  --predict-limit 10 --no-explain --seed 1
```
These are different measurements from the full reference.

## Interpret the output

`results.json` records environment, requested configuration, stage success, timing, realized batch size, model plan and metrics.
`run.log` and `model_plan.txt` preserve execution details.
Keep output directories separate; the runner refuses a nonempty output directory.
A stage failure exits nonzero and, after runtime initialization, records the stage error.
A reused snapshot is not an ingestion benchmark.
Check realized training steps and batch size because an OOM retry or later configuration override can differ from initial arguments.

The reference spent 3,640.7 seconds on snapshot preparation, 6,281.9 seconds on training-table generation and 8,661.2 seconds on fitting including validation/test and fit setup/shutdown.
Their sum is 5.16 hours, rounded to 5.2 on the website.
The separate five-million-customer prediction took 4,701.7 seconds inside prediction (78.4 minutes), or 5,201.8 seconds including setup (86.7 minutes).
It used four scorer processes with four loaders each.
Do not compare an extrapolated full-population estimate or a short steady-state probe with that completed batch duration.

The Kumo half of the feature-engineering comparison reuses this same commerce run.
The external DuckDB pipeline is not included; running that comparator requires software beyond Kumo's runtime dependencies.

## Reconstruct the feature-engineering baseline

The separate DuckDB feature-engineering comparison is a modeling pipeline, not the context sampler described by the context folder.
Its full scripts and extra libraries are omitted.
An agent rebuilding it should preserve the following recorded protocol and treat missing details as new implementation decisions, not recovered historical settings.

* Use the commerce graph's seven-day top-ten merchant recommendation query, timestamp fields, foreign keys and generated seed/anchor files.
  Construct each seed's labels from distinct merchants purchased after its anchor through anchor plus seven days; features must use only records available at the anchor.
  `scripts/make_eval_ids.py` defines the shared test population and targets for generated data.
* The archived October 11 DuckDB 1.5.5 pipeline requested 1,024,000 training seeds, 200,000 validation seeds and 1,000,000 test seeds, with 5,000,000 score customers.
  Training/validation seed selection code and all historical anchor assignments are not present in this package.
  Do not invent them and claim an exact reproduction; define and record a time-separated protocol for the generated-data comparison.
* Candidate sources were past purchases, regional popularity, co-purchase and co-view signals, with a recorded candidate cap K of 50.
  The final candidate union averaged 67.24 candidates per training/validation/test seed, so K was not a cap on the final union.
  Feature families included customer and merchant aggregates, buyer profiles, pair history and similar-customer purchases, including history windows such as 4, 13 and 52 weeks.
  Co-favorite and search feature groups were deliberately dropped in the archived run.
  Exact SQL expressions, window boundaries, source-specific candidate allocations and tie breaks are not recoverable from the results record alone.
* The model used XGBoost on CUDA, 106 features and a requested 500-round budget; the archived best iteration was 175.
  The full hyperparameter set is not included in the recovered record.
  Score rankings using MAP@10 and recall@10 on the identical test seed/target file, including seeds without predictions in the denominator.
  The recorded million-seed evaluation was MAP@10 0.5147 and recall@10 0.7601.
  The Kumo fit's 700-step test metric uses a different population and must not be presented as that same evaluation.
* Use one g6e.48xlarge per workflow, record the engine/runtime revisions, and separate source load, seed construction, merchant/customer features, candidates, pair features, labels, fitting, evaluation and batch scoring.
  The archived 9.885-hour figure sums source loading through fitting, excludes later evaluation/scoring, and includes source-load time recorded in the first run on that machine.
  Subsequent stages read the already staged local files.
  The reported Kumo 5.162-hour figure sums snapshot, training-table generation and fit, whose fit stage includes validation/test and setup/shutdown.
  These are workflow totals with different model budgets and stage boundaries, not matched training-loop speedups.

Rebuilding this comparator requires DuckDB and XGBoost in a separately managed baseline environment.
The generator and Kumo scripts in this folder do not import or install them.
A complete exact baseline reconstruction would require the missing SQL, split and model-parameter sources identified above.
