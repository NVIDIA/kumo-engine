# DiskGraph

DiskGraph is a Rust-based pipeline for turning CSV/Parquet tables into an on-disk graph index and exploring it with a fast TUI.

## Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Getting Started (Local, tested on GCP Local SSD)

### 1) Start a VM and mount Local SSD

We have a helper that formats and mounts local SSDs as `/mnt/localssd`. It is **destructive** and has only been tested on GCP so far.

```bash
sudo ./scripts/localssd-raid0-xfs-tui.sh
```

### 2) Download data to `/mnt/localssd/datasets/`

Use the H&M download helper (defaults to `/mnt/localssd/datasets/hnm`):

```bash
./diskgraph-datagenerator/scripts/hnm100_download.sh
```

### 3) Infer `source_metadata.json`

```bash
./diskgraph-infer/scripts/infer_source.sh /mnt/localssd/datasets/hnm
```

This writes `/mnt/localssd/datasets/hnm/source_metadata.json`.

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
./diskgraph-infer/scripts/infer_source.sh /mnt/localssd/datasets/small_100m
./diskgraph-ingest/scripts/ingest_local.sh /mnt/localssd/datasets/small_100m
```

## Download ingested graph data from a job

Use this when you have a training or bp job ID and want the raw parquet tables for ingest.

```bash
./diskgraph-datagenerator/scripts/download_ingested_graph.sh <customer_name> <job_id>
./diskgraph-infer/scripts/infer_source.sh /mnt/localssd/datasets/<customer_name>_<job_id>
./diskgraph-ingest/scripts/ingest_local.sh /mnt/localssd/datasets/<customer_name>_<job_id>
```
