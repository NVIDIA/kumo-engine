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

The `gxhash` dependency requires AES CPU intrinsics, so set `RUSTFLAGS` for
native builds:

```bash
export RUSTFLAGS="-C target-cpu=native"
```

### Quick build (no venv setup needed)

```bash
cd diskgraph-pybind

# Build a wheel
RUSTFLAGS="-C target-cpu=native" uvx maturin build --release

# Or install into a venv in one step
uv venv .venv
source .venv/bin/activate
RUSTFLAGS="-C target-cpu=native" uvx maturin develop --release
```

### Using an existing venv (e.g. kumo repo)

```bash
source /path/to/your/venv/bin/activate
RUSTFLAGS="-C target-cpu=native" uvx maturin develop --release --manifest-path diskgraph-pybind/Cargo.toml
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

```bash
cd diskgraph-pybind
./run_tests.sh
```

This builds the module with `uvx maturin develop` and runs pytest. Override the
Python interpreter with `PYTHON=python3.11 ./run_tests.sh`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_BATCH_SIZE_ROWS` | 65536 | Rows per edge batch sent to Python |
