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

Use the H&M download helper (defaults to `/mnt/localssd/datasets/hnm`):

```bash
./diskgraph-datagenerator/scripts/hnm100_download.sh
```

### 3) Infer `diskgraph_config.json`

```bash
./diskgraph-infer/scripts/infer_config.sh /mnt/localssd/datasets/hnm
```

This writes `/mnt/localssd/datasets/hnm/diskgraph_config.json`.

### 4) Ingest to a DiskGraph index

```bash
./diskgraph-ingest/scripts/ingest_local.sh /mnt/localssd/datasets/hnm
```

By default the output goes to `/mnt/localssd/diskgraph-index/hnm`.

### 5) Explore (and optionally benchmark)

```bash
./diskgraph-index/scripts/explore_local.sh /mnt/localssd/diskgraph-index/hnm
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
