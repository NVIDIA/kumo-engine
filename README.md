# DiskGraph

DiskGraph is a Rust-based pipeline for turning CSV/Parquet tables into an on-disk graph index and exploring it with a fast TUI.

## Getting Started (Local, tested on GCP Local SSD)

### 1) Start a VM and mount Local SSD

We have a helper that formats and mounts local SSDs as `/mnt/localssd`. It is **destructive** and has only been tested on GCP so far.

```bash
sudo ./scripts/localssd-raid0-xfs-tui.sh
```

### 2) Download data to `/mnt/localssd/datasets/`

Example with AWS S3:

```bash
mkdir -p /mnt/localssd/datasets/hnm
aws s3 cp "s3://kumo-public-datasets/Kaggle_H&M/parquet/articles.parquet" /mnt/localssd/datasets/hnm/
aws s3 cp "s3://kumo-public-datasets/Kaggle_H&M/parquet/customers.parquet" /mnt/localssd/datasets/hnm/
aws s3 cp "s3://kumo-public-datasets/Kaggle_H&M/parquet/transactions_train.parquet" /mnt/localssd/datasets/hnm/
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
