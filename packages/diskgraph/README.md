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

Monitor live disk throughput, latency, request sizes, queue depth, and
utilization during a run:

```bash
./scripts/diskstats.sh -d '^nvme'
```

## Generate a synthetic dataset

Use the JSON-driven datagenerator to write Parquet shards under `/mnt/localssd/datasets/<config-name>/`.

```bash
./diskgraph-datagenerator/scripts/generate_dataset.sh diskgraph-datagenerator/configs/small_100m.json
./diskgraph-infer/scripts/infer_config.sh /mnt/localssd/datasets/small_100m
./diskgraph-ingest/scripts/ingest_local.sh /mnt/localssd/datasets/small_100m
```

For cloud storage, pass your output location to the generator:

```bash
cargo run -p diskgraph-datagenerator --release -- \
  --config diskgraph-datagenerator/configs/small_100m.json \
  --output s3://example-bucket/diskgraph-datagenerator/small_100m
./diskgraph-infer/scripts/infer_config.sh s3://example-bucket/diskgraph-datagenerator/small_100m
./diskgraph-ingest/scripts/ingest_s3.sh s3://example-bucket/diskgraph-datagenerator/small_100m
```

## Download ingested graph data from a job

Use this when you have a training or bp job ID and want the raw parquet tables for ingest.

```bash
./diskgraph-datagenerator/scripts/download_ingested_graph.sh --bucket <bucket_name> <customer_name> <job_id>
./diskgraph-infer/scripts/infer_config.sh /mnt/localssd/datasets/<customer_name>_<job_id>
./diskgraph-ingest/scripts/ingest_local.sh /mnt/localssd/datasets/<customer_name>_<job_id>
```

Use `--customer-prefix` if the storage directory differs from the customer name.

## Python wheel builds

`.github/workflows/build.yml` builds the native Python wheel on every relevant
push and pull request. The wheel contains the extension and the standalone
`ingest` binary.

Kumo builds and distributes that artifact from its in-tree
`packages/diskgraph` mirror. DiskGraph does not publish independently versioned
or development wheels.

See `diskgraph-pybind/README.md` for local and portable Linux build commands.

The Python package build is self-contained from a source checkout. The PEP 517
backend builds and stages the standalone `ingest` binary before delegating to
maturin:

```bash
uv pip install .
```

The base package installs the native extension and bundled standalone ingest
binary. Use the `kumo` extra only when calling Kumo compiler APIs outside an
environment that already installs `kumo-api`.

## Contributing notes

### Error handling and logging conventions (required)

We standardized error handling and logging in [PR #140](https://github.com/kumo-ai/kumo-diskgraph/pull/140). Please keep all new work consistent with that contract.

#### 1) Prefer typed Rust errors over stringly errors

- Add/extend crate-local typed error enums (`thiserror`) for new failure modes.
- Avoid introducing ad-hoc `anyhow!(...)` or string errors at API boundaries when a typed variant is possible.
- Keep errors domain-specific (data/config/system) so callers can react correctly.

#### 2) Pybind must map Rust errors to the 3 Python exception classes

In `diskgraph-pybind/src/lib.rs`, route failures to:

- `DiskgraphDataError`
- `DiskgraphConfigError`
- `DiskgraphSystemError`

Use the existing mapping helpers/patterns (for TT-gen, engine, and `anyhow` classification) instead of introducing new generic Python exceptions.

#### 3) Use structured logging via `tracing`

- Use `tracing::{info, debug, warn, error}` with structured fields.
- Prefer context-rich logs at major boundaries (start/finish, row counts, durations, key IDs/paths).
- Keep log levels meaningful (avoid noisy `info` in tight loops).

#### 4) Ensure Rust logs are visible from Python runtime

Pybind uses `pyo3-log` integration so Rust logs flow into Python logging. New pybind entry points should follow the existing initialization and instrumentation style.

#### 5) Review checklist for PRs touching errors/logging

- [ ] New failure paths use typed Rust errors.
- [ ] Pybind maps failures to one of the 3 DiskGraph Python exceptions.
- [ ] No new generic `PyRuntimeError`/`PyValueError` in places where typed mapping should apply.
- [ ] Logs use structured `tracing` fields.
- [ ] Existing logging/error behavior is preserved unless intentionally changed and called out in PR description.

When in doubt, use PR #140 as the reference baseline.
