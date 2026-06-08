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
scripts/build_python_wheel.sh --python python3.10 --out target/wheels --zig
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
| `DiskGraphTTBuilder` | Low-level plan builder with methods: `temporal()`, `for_each()`, `node_prediction()`, `link_prediction()`, `aggregate()`, `column()`, `compare()`, `filter()`, `split_ratio()`, `build()`. |
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
