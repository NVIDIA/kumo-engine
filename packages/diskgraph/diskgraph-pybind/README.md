# diskgraph-pybind

PyO3 native Python module wrapping the `DiskGraphEngine` directly, bypassing
Arrow Flight for co-located (same-machine) sampling and feature fetching.

## Prerequisites

- **Rust toolchain** (stable)
- **Python 3.7+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Building

The `gxhash` dependency requires AES CPU intrinsics. The workspace
`.cargo/config.toml` sets portable x86_64 and aarch64 flags by default. If
you override `RUSTFLAGS`, keep the target features explicit:

```bash
# x86_64
export RUSTFLAGS="-C target-feature=+aes,+sse2"

# aarch64
export RUSTFLAGS="-C target-feature=+aes,+neon"
```

Do not use `-C target-cpu=native` for wheels or Kumo-deployed builds. It can
embed host-specific instructions such as AVX-512 into the standalone ingest
binary packaged at `diskgraph/bin/ingest`, causing `SIGILL` on workers with a
different CPU. Package builds reject `target-cpu=native`.

### Quick build (no venv setup needed)

```bash
# From the kumo-diskgraph repository root:

# Build a wheel through the package backend. This also stages
# diskgraph/bin/ingest into the wheel.
python -m pip wheel --no-deps --wheel-dir dist .

# Or install into a venv in one step
uv venv .venv
source .venv/bin/activate
uv pip install .
```

The base install includes the native extension and bundled standalone
`diskgraph/bin/ingest` binary. Install `.[kumo]` only when using the
`DiskGraphPlanCompiler` APIs outside a Kumo environment that already provides
`kumo-api`.

Portable Linux wheels should be built through the repository wrapper:

```bash
# From the kumo-diskgraph repository root:
scripts/build_python_wheel.sh --python python3.11 --out target/wheels --zig
```

`--zig` builds the Python extension and bundled `diskgraph/bin/ingest` binary
against manylinux2014 so the resulting Linux wheel is not tied to the build
host's glibc. The wrapper installs `cargo-zigbuild` when needed; pip build
isolation installs the Python build requirements from `pyproject.toml`.

### Using an existing venv (e.g. kumo repo)

```bash
# From the kumo-diskgraph repository root:
source /path/to/your/venv/bin/activate
uv pip install .
```

## Usage

```python
import json
import diskgraph_pybind

# Load engine from disk index
engine = diskgraph_pybind.SamplerEngine("/path/to/index")

# Schema discovery
mapping = json.loads(engine.get_type_mapping())
print(mapping["node_types"])
print(mapping["edge_relations"])

# Sample (topology only)
plan = {"layers": [{"fanouts": {"rel_name": {"fanout": 10, "mode": "uniform",
        "absent_key_policy": "uniform_fallback"}}, "default_fanout": None}]}
for event in engine.sample({"node_type": [0, 1, 2]}, json.dumps(plan), fused=False):
    print(event["type"], event.get("rel_id") or event.get("type_id"))

# Sample with fused features
for event in engine.sample({"node_type": [0, 1, 2]}, json.dumps(plan), fused=True):
    if event["type"] == "feature":
        print(event["batch"].num_rows, "feature rows for type", event["type_id"])

# Standalone feature fetch
features = engine.fetch_features([
    {"node_type": "users", "node_ids": [0, 1], "feature_names": []}
])
for event in features:
    print(event["batch"].schema)
```

## Running tests

Python tests:

```bash
uv pip install ".[test]"
TEST_DIR="$(pwd)/diskgraph-pybind/python/tests"
cd /tmp
pytest -v "$TEST_DIR"
```

This installs the package through the same backend used by Git URL installs, so
the standalone `ingest` binary is bundled before tests run.

Rust unit tests (`diskgraph-pybind` crate):

```bash
cargo test -p diskgraph-pybind
```

The crate keeps `extension-module` as an opt-in feature. Maturin enables it for
Python extension builds, while plain Rust tests run without it to avoid Python
C API linker issues in test binaries.

## Training Table Generation API

In addition to sampling/features, `diskgraph-pybind` exports the TT generation pipeline:

### Python Classes

| Class | Description |
|-------|-------------|
| `DiskGraphPlanCompiler` | Compiles a Kumo `ValidatedPredictiveQuery` AST into a DiskGraph execution plan (JSON). |
| `DiskGraphTTBuilder` | Low-level plan builder with methods: `temporal()`, `for_each()`, `node_prediction()`, `link_prediction()`, `aggregate()`, `column()`, `compare()`, `filter()`, `split_ratio(spec)`, `build()`. `split_ratio` takes a `SplitSpec` (or any object carrying its attributes) as a single argument. |
| `TTPlanIterator` | Streaming iterator over RecordBatches from the Rust TT gen executor. Implements `__iter__`/`__next__`. Releases GIL during `recv()`. |
| `EdgeRef` | Directed FK reference: `(src_table, src_col, dst_table, dst_col)`. |
| `RowFilter` / `StageFilter` | Filter specifications for per-row and per-stage filtering. |
| `SplitSpec` | Split configuration: ratio, kind, offsets, ranges, temporal flag. |

### Python Functions

| Function | Description |
|----------|-------------|
| `execute_tt_plan(plan_json, index_path)` | Execute a TT plan and return a `TTPlanIterator` for streaming results. |
| `execute_tt_plan_with_artifacts(plan_json, index_path, output_path, ...)` | Execute TT plan with full artifact production (parquet, splits, stats, edges, manifest). |
| `export_tt_artifacts(export_config)` | Export edge artifacts (fkey_cols, reverse edges) from a materialized TT. |
| `compute_dense_split_assignments(split_config, ...)` | Compute split assignments for a TT. |
| `ingest(config_json, output_path, ...)` | Run ingest from Python. |
| `convert_kumo_config(kumo_config, ...)` | Convert Kumo GraphConfig to DiskGraph ingest config. |
| `estimate_ingest_size(config_json)` | Estimate output size before ingest. |

### Config Converter

`DiskgraphConfigConverter` converts Kumo's `GraphConfig` (SDK-level graph definition) to DiskGraph's ingest config:

```python
from diskgraph import DiskgraphConfigConverter

converter = DiskgraphConfigConverter(kumo_graph_config, data_persist_dir, table_data_versions)
diskgraph_config = converter.convert()
```

Handles table type detection, FK resolution, timestamp format inference, num_shards estimation, and source path expansion.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_BATCH_SIZE_ROWS` | 65536 | Rows per edge batch sent to Python |

### Embedded native sampling server

`diskgraph.SamplingServer` starts the Rust Flight gateway on native background
threads inside the Python process. No sampling subprocess is required. The caller
must materialize and verify training-table extensions beneath the local graph
before requesting epochs; the sampling transport only reads prepared artifacts.

```python
import diskgraph

with diskgraph.SamplingServer(
    "/local/graph", host="0.0.0.0", max_streams=4, max_threads=2
) as server:
    endpoint = server.endpoint("10.0.0.5")
    # Keep the handle alive while remote trainers use this endpoint.
```

`port` reports the actual bound port; `health()` reports native counters and
readiness. `close()` cancels streams and joins the listener while releasing the
GIL. Dropping a handle signals asynchronous shutdown; use explicit close or the
context manager for deterministic cleanup. Configure Python logging before
starting the server to receive native lifecycle and progress events.

`diskgraph.SamplingClient(endpoint).open(request)` accepts an immutable native
`SamplingRequest`, serializes its Flight ticket in Rust, and returns a `SamplingStream` with `num_samples`, `num_batches`, iteration and
`close()`. The Rust client accepts component schema changes and exports complete
native batches through the Arrow C Data Interface. Column references return
arrays; column-range references return record batches, including grouped targets. Closing a client cancels its
streams; closing a stream interrupts blocked reads. Network reads release the
GIL. The standard PyArrow Flight reader assumes one schema and cannot consume
this component protocol. See the [gateway contract](../diskgraph-gateway/README.md).


```python
request = diskgraph.SamplingRequest(
    "train_tables/churn",
    stage_table="train_seeds",
    sampling_plan_json='{"layers":[]}',
    batch_size=512,
    epoch=0,
    engine_seed=42,
    num_threads=1,
)
client = diskgraph.SamplingClient(endpoint)
try:
    for batch in client.open(request):
        consume(batch)
finally:
    client.close()
```

The native constructor accepts keyword iterator settings, validates batch/thread
budgets and shard bounds, and owns the request kind and protocol version.
`open()` accepts only a `SamplingRequest`; Python does not construct a wire
dictionary or serialize ticket bytes. Sampling-plan and weighted-sampling JSON
strings retain the existing native iterator interface. Requests can be reused
without cloning their native seed-selection arrays during serialization.


### Shared local and remote runtime interface

The public manifest wrapper uses composition to select its native backend:

```python
index = diskgraph.RuntimeIndex(
    "runtime_manifest.json",
    rnd_seed=42,
    num_threads=1,
    sampling_endpoints=["grpc://worker-a:8815", "grpc://worker-b:8815"],
)
try:
    batches, count = index.iter(
        stage="train", sampling_plan={"layers": []},
        num_workers=1, worker_id=0, shuffle=False, epoch=0, batch_size=512,
    )
    try:
        for batch in batches:
            consume(batch)
    finally:
        batches.close()
finally:
    index.close()
```

Omitting `sampling_endpoints` uses the existing local native index. Both paths
share manifest resolution, argument normalization and minibatch export. Local
sampling and the server convert native batches into `diskgraph_runtime::ArrowBatch`;
the remote client receives that same representation. One Python exporter restores
the task/graph/feature structure through the Arrow C Data Interface. Local
iteration builds components in memory without IPC or JSON serialization or a
transport byte limit. The remote backend owns
connections, epoch sharding, one-batch native prefetch, round-robin merging and
cancellation; Python only exports the selected complete batch. Closing the remote
epoch leaves its index available for later stages/epochs. Closing the index is
terminal. The lower-level `diskgraph.RemoteRuntimeIndex(endpoints)` class accepts
`iter(SamplingRequest, graph_root)` directly and returns `RemoteBatchIterator`.
Both handles expose `close()`; garbage collection signals asynchronous cleanup
without joining while holding the GIL.


All iterators returned by `RuntimeIndex.iter()` expose an idempotent `close()`.
Remote iteration cancels pending readers and joins its prefetch threads. Local
iteration is synchronous and implements a no-op close; normal iterator/index
ownership releases its resources. Callers can register `iterator.close` directly
for scope cleanup, without probing for an optional method.
