# Commerce context generation

Measure temporal neighborhood sampling and raw feature retrieval on the synthetic commerce graph.
The accompanying reference records are the completed October 21, 2026 trial on engine commit `392c25ed2439e0eb2f8c3722192e6e1aff783ad6`.
The documented comparison covers two, three and four hops.

## Contents and dependencies

`scripts/` in the downloadable folder contains the sampling harness, evaluation seed preparation, and the shared commerce graph/ingest runner.
In the repository, the sampling scripts are beside this README; the graph and ingest runner are in `benchmark/commerce/`.
The download preserves these scripts without requiring a repository checkout.
`reference-results.json` records the measured configuration and rates, including the external DuckDB comparison.
The runnable scripts execute Kumo only.
They use Python's standard library, NumPy, PyArrow and the installed Kumo runtime; they do not install packages or require DuckDB, AWS CLI, or pytest.
Install a compatible Kumo Engine runtime following the Engine setup guide before running them.
The reference used Linux x86_64 and Python 3.12.3.

## Data and hardware

Provide a local commerce rung with seven table directories: `customers`, `merchants`, `purchases`, `merchant_views`, `searches`, `favorites`, and `reviews`.
Each directory contains Parquet parts.
The measured 50B rung contains 49,996,770,873 rows, including 1,392,680,000 customers, in 497.93 GiB of Parquet.
It requires the original `_eval/test_seeds.parquet` to reproduce the measured seed population.
Use the included generator for a representative dataset.
Generated data and its derived evaluation seeds are not the original measured inputs.

The reference host was one i4i.32xlarge with 128 vCPUs, 1 TiB nominal RAM and eight 3.75 TB local NVMe disks striped together.
The index occupied 2,635 GiB and the ingest process peaked at about 647 GiB RSS.
Use local SSD scratch and size capacity for both the source and index.
A 100M rung can exercise the workflow before a full run, but its results do not reproduce the 50B measurement.

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

Run from the extracted folder and replace the three paths with your local directories.
Use a new output directory for each measurement and a fresh scratch directory when timing ingestion.

```bash
DATA=/path/to/commerce/50B
SCRATCH=/path/to/local-ssd/context-scratch
OUT=/path/to/context-results
python scripts/run_e2e.py --graph commerce --data "$DATA" --scratch "$SCRATCH" --out "$OUT/ingest" --stages snapshot --num-workers 0
```

Read `stages.snapshot.index_path` from `$OUT/ingest/results.json` and use it as `INDEX` below.
Snapshot reuse is reported as `reused: true`; it is not a new index-build measurement.

```bash
INDEX=/path/printed/in/results.json
python scripts/dg_bench.py map --index "$INDEX" --seeds "$DATA/_eval/test_seeds.parquet" --n 200000 --threads 128 --out "$OUT/seedmap.parquet"
python scripts/run_matrix.py --rung 50B --index "$INDEX" --seedmap "$OUT/seedmap.parquet" --vcpus 128 --dg-threads 8 --hops 2,3,4 --out "$OUT/results.jsonl"
```

The command requests file-level page-cache eviction before the cold pass.
The reference trial used whole-host cache eviction.
To request that exact cache procedure, add `--drop-all` only on an isolated host with permission to write `/proc/sys/vm/drop_caches`.
The script fails if that operation is denied.
A warm pass repeats the same seed sequence immediately afterwards; it does not imply the 2.6 TiB index fits in RAM.
The map contains 200,000 seeds and the harness repeats it when a cell requests more seeds.
Counts therefore refer to seed evaluations, not necessarily distinct customers.

For a small local rung without evaluation seeds, generate them first:

```bash
python scripts/make_eval_ids.py --root "$DATA" --out "$DATA/_eval" --test-seeds 100000 --workers 4
```

This derives seeds from that rung and is suitable for a smoke test, not replacement of the published 50B evaluation files.

## Read the results

Each JSONL row records depth, cache state, requested work, completed seeds, rates and status.
Use only `status: ok` cells as completed results.
`seeds_per_s` divides completed seeds by the slowest worker's sampling interval, excluding worker initialization and coordinator completion overhead.
`wall_s` is a separate coordinator timer and need not equal `seeds / seeds_per_s`.
Graph ingestion, customer-ID mapping and model training are excluded from the retrieval rate.
The website's normalized seconds per million are calculated as `1000000 / seeds_per_s`; they are not measured million-seed end-to-end durations.

The plan uses the prefix of `24,8,6,8,6,8`, temporal-most-recent sampling at each seed's anchor, both directions of each foreign key, and raw feature retrieval.
The two-to-four-hop reference uses 16 processes with eight threads each, batches of 512/256/128 and 1,000,000/300,000/200,000 seed evaluations.
Correctness checks established identical neighborhoods for 300 seeds each at two and three hops.
No four-hop equality result is included in this reference.
External comparator records document the historical trial; the downloadable scripts do not execute DuckDB or Polars.

Missing input, invalid dimensions, a failed worker, or an unwritable cache control should be resolved before interpreting output as a completed trial.
Preserve the runtime revision, data hashes and cache procedure alongside every new result.

## Reconstruct the external DuckDB comparison

The comparison implementation is intentionally omitted so the download adds no DuckDB dependency.
An agent implementing it should use the following contract, then install DuckDB separately in its comparator environment.
Do not report regenerated-data results as reproducing the archived data's exact rates.

1. Read the schema in `generator/scalebench/schema.py` and the generated manifest.
   `customers.customer_id` and `merchants.merchant_id` are entity keys.
   `purchases`, `merchant_views`, `favorites` and `reviews` each reference both entities; `searches` references customers only.
   Use each table's declared timestamp and raw columns; assign stable physical-row IDs to event tables without a unique declared key.
2. Reuse `scripts/make_eval_ids.py` and exactly the same ordered seed/anchor file for both implementations.
   It finds the data's largest timestamp T, anchors test seeds at T minus seven days, selects customers signed up by the anchor with a purchase in (anchor, T], and ranks candidates with the specified portable hash.
   `np_hash`, the salts and the customer-ID tie break in that script are the executable selection specification.
   Match repeated seed evaluations and per-system cell sizes in `reference-results.json`; do not replace them with a different unique-customer sample.
3. Start each subgraph with its seed customer, expand both directions of all nine foreign keys, and retain the seed's anchor through every hop.
   For an entity-to-event relation, retain events with event time at or before the anchor and take the most recent k per frontier node and relation, with fanout prefix `24,8,6,8` for the displayed depths.
   For an event-to-entity relation, follow its foreign key.
   The recorded SQL form is `max_by(event_id, event_time, k)` grouped by frontier ID and anchor.
   Do not deduplicate across seeds: deduplicate nodes by `(seed, node_type, node_id)` against all earlier hops within that seed.
   Fetch the complete raw row for every retained node, including seeds, inside the timed operation.
4. Equal timestamps at the kth boundary can select different valid rows.
   The timed `max_by` implementation does not promise the same tie order as DiskGraph.
   For correctness, compare sorted per-seed/per-type/per-hop row content, using `common.KEY` and `dg_bench.py dump`; separately count tie-free mismatches and tied seeds.
   If exact tied-row equality is required, obtain DiskGraph's stored ordering for the tied boundary and use it only in the correctness comparator, recording that extra condition.
   Do not claim timing runs universally matched because 300 correctness seeds matched.
   Destination-time filtering on event-to-entity traversal was an optional source setting; verify its actual behavior against the Kumo dump on adversarial future-created parents before claiming equivalence on a new dataset.
5. Build the prepared variant from native on-disk DuckDB tables plus adjacency tables ordered by foreign key and timestamp.
   The direct-Parquet variant reads source Parquet without that one-time materialization.
   Report preparation independently: the archived native load/sort took 2,313.02 seconds.
   The recorded comparator used DuckDB 2.0.0 on an i4i.32xlarge, eight processes with 16 threads each, and batches of 100,000 / 50,000 / 20,000 at two / three / four hops.
6. Drop caches before the cold pass on an isolated host, then repeat the identical seeds for warm.
   Synchronize worker start; measure from the released start barrier through sampling and raw-row fetch in each worker.
   Divide all completed seeds by the slowest worker interval.
   Also retain coordinator wall time, first-batch time, status, rows returned, requested/completed seeds, process/thread counts and preparation time.
   Normalize seconds per million as `1e6 / seeds_per_s`, keeping that derived number distinct from measured elapsed time.
   Publish successful complete cells with hardware, software, data/config hashes and cache procedure, and show incomplete cells as incomplete.

The existing Polars GPU figures on the website are a separate prior report.
The matching raw GPU artifact has not been recovered for this package; CPU streaming Polars records are not a replacement for it.
