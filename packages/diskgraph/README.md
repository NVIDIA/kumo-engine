# DiskGraph

DiskGraph is a Rust-based pipeline for turning CSV/Parquet tables into an on-disk graph index and exploring it with a fast TUI.

## Sampling I/O configuration

Set these environment variables before opening an index or starting training,
prediction, or DataLoader workers. Existing readers retain their backend, and
the I/O runtime and direct-cache settings initialize once per process. Restart
the relevant processes to change their settings.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DG_EDGE_BACKEND` | `mmap` | `mmap` uses the existing memory-mapped reader. `buffered` submits buffered reads through Linux `io_uring`; `direct` uses Linux `io_uring` with `O_DIRECT`. Names are case-sensitive; unknown values are rejected. |
| `DG_DIRECT_CACHE_GB` | `0` | Nonnegative integer GiB of cached page data for `direct` mode only; `0` disables the fixed page cache. One GiB is `2^30` bytes. Actual data capacity is rounded to the cache geometry and reported at initialization; metadata uses additional memory. |
| `DG_IO_RINGS` | `8` | I/O owner threads/rings per process for explicit reads, clamped to `1..64`. |
| `DG_IO_QD` | `256` | Queue depth per I/O owner, clamped to `1..4096`. |

Use decimal integers for numeric settings. Invalid numeric strings use their
defaults; this differs from an invalid backend name, which raises an error.

The fixed direct cache is shared by readers and I/O owners **within one
process**, not across processes. Budget the sum of the configured caches across
all processes that perform direct reads, including each active DataLoader
worker and any parent process that reads. For example, four workers configured
with `DG_DIRECT_CACHE_GB=1` can use about four GiB of cached page data in total,
plus cache metadata and other training/I/O memory. Pages populate on demand;
the configured data capacity is not an immediate resident-memory allocation.

`buffered` and `direct` require Linux with working `io_uring`; `direct` also
requires `O_DIRECT` support from the storage/filesystem. Requesting either
backend on macOS or native Windows returns an unsupported-operation error
(`DiskgraphSystemError` in Python). A permission or I/O failure is reported;
there is no automatic retry using `mmap`.

Automatic native sampling and Core training configuration select standalone
feature fetching for `buffered`/`direct`. An explicit native `fused=True`
request is unsupported for these backends and raises `DiskgraphConfigError`.
Leave native `fused` selection automatic or pass `False` when selecting an
explicit backend.

This Linux example uses settings exercised by the correctness tests:

```sh
export DG_EDGE_BACKEND=direct
export DG_DIRECT_CACHE_GB=1
export DG_IO_RINGS=2
export DG_IO_QD=8
# Start the training or prediction process with this environment.
```

These controls select the I/O path and resource budget. Correctness checks do
not establish a throughput improvement; compare representative workloads
before changing deployment defaults.

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

The cached-O_DIRECT sampler design, correctness gate, fio ceiling, and retained
CPU/throughput baselines are recorded in
the [benchmark report index](docs/reports/README.md).

### Benchmark sampling IOPS

Use the production-engine sweep to measure nodes/s, physical read IOPS, and
average queue depth with a stable memory cap and per-index cache eviction:

```bash
cargo build --release -p diskgraph-engine --bin dg-engine-bench --features jemalloc
./scripts/bench_sampling_iops.sh \
  /mnt/localssd/diskgraph-index/alt_detection_full_scale-zstd-1c1f7a9f-full-monitor
```

The default sampling workload starts at `users` and traverses three hops with
fanout 10 on every applicable edge type:

```text
users -> {ip_addresses, mac_addresses, emails, btids}
      -> users
      -> {ip_addresses, mac_addresses, emails, btids}
```

Its canonical plan digest is `43d5badeb6e6f065`. The default frontier batch
is 512 roots; this workload produces about 1,514 output nodes/root on the
alt-detection indexes. Set `DG_BENCH_PLAN` to override the plan explicitly.

Use the read-only fio helper to establish the storage ceiling. Raw targets
require an explicit acknowledgement and block-device read permission:

```bash
./scripts/bench_fio_iops.sh /path/to/large/index-file
FIO_ALLOW_RAW_READ=1 ./scripts/bench_fio_iops.sh /dev/md127
```

See the
[`i8g identity scale matrix`](docs/reports/diskgraph-i8g-identity-scale-matrix-2026-08-10.md)
for the current-hardware 1%, 5%, 25%, and 50% mmap/Direct comparison, including
the additive Threadripper comparison. Cached O_DIRECT reaches 404.096M nodes/s
at 1%, 290.582M at 25%, and 105.324M while driving the six-SSD array to 92.3%
of its fio ceiling at 50%. The earlier
[`Threadripper scale matrix`](docs/reports/diskgraph-all-edge-3hop-scale-matrix-2026-08-08.md)
retains the 1%, 5%, 25%, 50%, and 100% results from that different host. At
full scale there, cached O_DIRECT reaches 104.437M nodes/s and 6.123M physical
IOPS versus mmap's 22.611M nodes/s and 1.306M IOPS: a 4.619x throughput gain
while using 67.6% fewer total CPU-seconds. The
[`detailed full-scale report`](docs/reports/diskgraph-all-edge-3hop-100pct-mmap-vs-direct-2026-08-07.md)
contains the adjacent profiler, allocation, cache, and runtime-gap analysis.
The separate
[`Pinterest i8g report`](docs/reports/diskgraph-pinterest-i8g-io-benchmark-2026-08-09.md)
records paired pruned/unpruned OmniSage results on a 96-core Graviton4 host with
742 GiB RAM and six Nitro SSDs. The additive
[`i8g feature-heavy matrix`](docs/reports/diskgraph-i8g-ecommerce-feature-io-benchmark-2026-08-10.md)
uses a steady 640-GiB Direct cache and records 2.390x and 2.267x Direct gains
on the out-of-core 25% and 50% e-commerce graphs, alongside the original
Threadripper measurements. Other sampler reports describe the earlier
two-hop IP-only workload and are not directly comparable.

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

The repository's CI builds the native Python wheel for every relevant push and
pull request (`.github/workflows/_diskgraph-wheel.yml`), and the kumo-engine
wheel bundles it. The wheel contains the extension and the standalone `ingest`
binary.

Kumo builds and distributes that artifact from its in-tree
`packages/diskgraph` source. This repository is the sole source of truth for
DiskGraph; make changes here directly. DiskGraph does not publish independently
versioned or development wheels.

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

## Native batch runtime

[`diskgraph-runtime`](diskgraph-runtime/README.md) provides reusable index ownership
and worker-sharded batch generation without Python or transport dependencies. The
Python binding uses this runtime and converts its batches to Arrow-backed Python
objects.

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
