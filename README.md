# DiskGraph

DiskGraph is a Rust-based pipeline for turning CSV/Parquet tables into an on-disk graph index and exploring it with a fast TUI.

## Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Getting Started (Local, tested on GCP Local SSD)

### 1) Start a VM and mount Local SSD

We have a helper that formats and mounts local SSDs as `/mnt/localssd` (override with `DISKGRAPH_DATA_ROOT`). It is **destructive** and has only been tested on GCP so far. If you do not have attached NVMe devices (e.g., on a laptop), set `export DISKGRAPH_DATA_ROOT="$HOME/diskgraph-data"` and skip the local SSD setup.

```bash
sudo ./scripts/localssd-raid0-xfs-tui.sh
```

### 2) Download data to `/mnt/localssd/datasets/`

If you want to use a different base directory, set `DISKGRAPH_DATA_ROOT=/path/to/data` and keep the rest of the paths the same.

Use the H&M download helper (defaults to `/mnt/localssd/datasets/hnm100`):

```bash
./diskgraph-datagenerator/scripts/hnm100_download.sh
```

### 3) Infer `diskgraph_config.json`

```bash
./diskgraph-infer/scripts/infer_config.sh /mnt/localssd/datasets/hnm100
```

This writes `/mnt/localssd/datasets/hnm100/diskgraph_config.json`.

### 4) Ingest to a DiskGraph index

```bash
./diskgraph-ingest/scripts/ingest_local.sh /mnt/localssd/datasets/hnm100
```

By default the output goes to `/mnt/localssd/diskgraph-index/hnm100`.

### 5) Explore (and optionally benchmark)

```bash
./diskgraph-index/scripts/explore_local.sh /mnt/localssd/diskgraph-index/hnm100
./diskgraph-ingest/scripts/perf_explorer.sh hnm100
```

### 6) Run predictive queries (train table TUI)

```bash
./diskgraph-train-table/scripts/train_table.sh /mnt/localssd/diskgraph-index/hnm100
```

Inside the explorer:

```
tables
use <table>
info
node <id>
benchmark 8 30s
```

`benchmark` is optional but handy to sanity-check performance.

### Benchmark ingest runs (S3-driven)

We have a server running 24/7 for ingest benchmarks. To add a dataset to the benchmark rotation, update `scripts/benchmarks/benchmark_configs/config_list.txt`.

```bash
./scripts/benchmarks/benchmark_ingest_systemd.sh install
./scripts/benchmarks/benchmark_ingest_systemd.sh status
./scripts/benchmarks/benchmark_ingest_tail.sh
./scripts/benchmarks/benchmark_ingest_systemd.sh stop
./scripts/benchmarks/benchmark_ingest.sh
./diskgraph-ingest/scripts/perf_explorer.sh
```

## Generate a synthetic dataset

Use the JSON-driven datagenerator to write Parquet shards under `/mnt/localssd/datasets/<config-name>/`.

```bash
./diskgraph-datagenerator/scripts/generate_dataset.sh diskgraph-datagenerator/configs/small_100m.json
./diskgraph-infer/scripts/infer_config.sh /mnt/localssd/datasets/small_100m
./diskgraph-ingest/scripts/ingest_local.sh /mnt/localssd/datasets/small_100m
```

Cloud example (auto-detects AWS/GCP/Azure metadata, writes to `kumo-unit-test` by default):

```bash
./diskgraph-datagenerator/scripts/cloud_generate_dataset.sh diskgraph-datagenerator/configs/small_100m.json
./diskgraph-infer/scripts/infer_config.sh s3://kumo-unit-test/diskgraph-datagenerator/small_100m
./diskgraph-ingest/scripts/ingest_s3.sh s3://kumo-unit-test/diskgraph-datagenerator/small_100m
```

## Download ingested graph data from a job

Use this when you have a training or bp job ID and want the raw parquet tables for ingest.

```bash
./diskgraph-datagenerator/scripts/download_ingested_graph.sh <customer_name> <job_id>
./diskgraph-infer/scripts/infer_config.sh /mnt/localssd/datasets/<customer_name>_<job_id>
./diskgraph-ingest/scripts/ingest_local.sh /mnt/localssd/datasets/<customer_name>_<job_id>
```

## Python wheel CI / nightly publishing

The repo now has two wheel-related workflows:

- `.github/workflows/build.yml`
  - Runs on push/PR to `main`
  - Builds the Python wheel from `diskgraph-pybind` using `maturin`

- `.github/workflows/nightly.yml`
  - Runs daily at `0 6 * * *` UTC and via manual dispatch
  - Builds wheels for:
    - `ubuntu-20.04`, Python `3.10`
    - `ubuntu-22.04`, Python `3.10`
  - Uploads wheels to S3:
    - `s3://kumo-diskgraph/packages/<os>/py<python>/latest/<wheel-file>`
    - `s3://kumo-diskgraph/packages/<os>/py<python>/<version>.dev<YYYYMMDD>/<wheel-file>`

Required GitHub Actions secrets for nightly upload:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

Region used by the workflow: `us-west-2`.

Current Python package version is read from `diskgraph-pybind/pyproject.toml` (`[project].version`).

