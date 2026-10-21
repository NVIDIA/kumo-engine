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

`lookup_node_ids` and `canonical_node_mask` parse and query one key at a time
within the supplied Arrow batch. Neither retains a second batch of parsed keys;
only the returned IDs or booleans accumulate.

```python
import json
import diskgraph_pybind

# Load engine from disk index
engine = diskgraph_pybind.SamplerEngine("/path/to/index")

# Schema discovery
mapping = json.loads(engine.get_type_mapping())
print(mapping["node_types"])
print(mapping["edge_relations"])

# Resolve external/raw primary keys before entering the dense-ID sampler.
# Missing, well-formed keys preserve position as None.
import pyarrow as pa
root_ids = engine.lookup_node_ids("users", pa.array(["u1", "missing", "u2"]))
print(root_ids)

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

### Direct temporal rows

`SamplerEngine.query_temporal(seed_nodes, seed_timestamps, plan_json,
output_format=...)` is the small co-located boundary for clients that already
own their sample table. It accepts one dense root type, aligned epoch-second
cutoffs, and a JSON plan with:

- `frames`: named exact relation paths, an optional relative time window, row
  distinctness, and the union of columns needed by consumers;
- `sequences`: ordered/limited projections of a frame; and
- `aggregates`: reductions over a frame, optionally after one or more nested
  grouping stages.

Frames are the unit of shared work. A sequence and several aggregates naming
the same frame perform one traversal and one rowpack decode. Results include
Arrow sequence batches, aligned offsets/timestamps, scalar aggregate values,
and nanosecond counters for traversal, decode/selection, sequence assembly,
aggregation, and the full native call. `stats.rows_decoded` counts rows sent to
the rowpack decoder. When the final relation has a matching lossless timestamp
sort key, bounded frames discard out-of-window edges before decoding their row
attributes; other relations retain the rowpack timestamp path. This API does
not build training tables, fit encoders, tokenize values, or choose model layouts.

For a one-hop, time-windowed frame with one bounded sequence and no aggregate
consumer, a matching lossless indexed clock allows selection before feature decoding.
The timestamp boundary retains all ties; the existing primary-key ordering and
sequence limit choose the final rows. Distinct frames count distinct node IDs
when choosing that boundary. If a candidate reaches the encoded timestamp
ceiling, selection falls back to full-frame decoding: ingestion may have clamped
its real timestamp. Other frames also retain full-frame decoding. This
reduces rowpack work, not edge scanning: candidate edge buffers still grow with
the adjacency, and boundary ties can require decoding more than the limit.

When one bounded sequence shares a frame with aggregates, traversal is still
shared. The full frame decodes only its keys, clock, and aggregate input columns,
preserving the existing filtering, ordering, and reductions. Additional sequence
columns are decoded only after selecting its final rows; shared columns reuse
the full frame's values. To avoid a costly second pass for low-selectivity
histories, this strategy requires the estimated selection to be at most one
quarter of the frame's candidates across the batch; otherwise decoding stays
single-pass. This is a conservative performance heuristic, not a different
answer or refusal. It does not combine different frames or time windows.
`stats.values_decoded` counts rows times projected columns across decoding calls;
`rows_decoded` can increase because selected rows are revisited for missing
columns. Deferred decoding is included in `timings_ns.sequence_assembly`.

`output_format="inspectable"` is the compatibility default: offsets, node IDs,
timestamps, and aggregate vectors are Python lists. `output_format="model"`
returns those vectors as Arrow arrays as well, avoiding one Python object per
value when a caller will immediately encode a batch. Sequence `batch` values
are Arrow `RecordBatch` objects in both formats. Query semantics, ordering,
limits, reductions, nulls, timings, and statistics are identical.

`SamplerEngine.query_temporal_lengths(seed_nodes, seed_timestamps, plan_json)`
is the exact scheduling companion. It visits only frames named by sequences,
shares a frame traversal between all of its sequence consumers, applies each
sequence limit, and returns one Arrow `uint64` length vector per sequence. It
does not build output batches, decode feature columns, sort selected rows, or
materialize aggregates. When the final relation's destination sort-key clock
matches the frame clock at lossless one-second resolution, relative-window
filtering decodes the timestamp already packed beside each edge. Otherwise it
decodes only the destination rowpack's timestamp column. `timings_ns` separates
traversal and filtering;
`stats` reports which timestamp path ran.

### Flat token packing for RDL

`diskgraph.pack_token_stream(event_offsets, parts, max_tokens, keep_last,
event_boundary, separator_id, separator_between, bos_id, eos_id)` assembles
already encoded Arrow `int64` columns. `event_offsets` delimits each sample's
events; each `(event_ids, token_ids)` part supplies one token column or ragged
text tokens. The result is `(sample_offsets, token_ids)`, both Arrow `int64`
arrays. Each sample is kept separate, separators and optional BOS/EOS are
inserted, and truncation respects the declared token budget and event-boundary
policy. Invalid or null buffers raise `DiskgraphConfigError`.

The packing loop runs in Rust with the Python GIL released. This primitive does
not depend on the graph index or store vocabulary state: RDL obtains temporal
Arrow columns through `query_temporal`, encodes them in columns, and calls the
packer to interleave tokens into model-ready streams. Python tests cover ragged
parts, sample isolation, both truncation directions, boundaries, and malformed
buffers. The companion RDL integration compares the complete streams with its
record-based reference.

### RDL model-boundary decision

The RDL integration builds direct entity fields, fitted graph node features,
shared categorical/numerical/text field tokens, nested event layouts, calendar
and elapsed-time arrays, masked-value targets, and token-aware batches on top of
the typed result. Those transformations deliberately remain client-owned:

- DiskGraph resolves dense IDs, walks the index, applies per-root temporal
  selection, reduces frames, and returns Arrow values.
- RDL defines and persists vocabulary IDs, numeric boundaries, BPE state,
  layouts, masking, and the final tensor contract.

The plan is intentionally small and explicit. Relation names and destination
types are physical names returned by `get_type_mapping()`:

```python
plan = {
    "frames": [{
        "name": "orders",
        "root_type": "users",
        "path": [{
            "relation": "orders_user_id_to_users_user_id_reverse",
            "destination_type": "orders",
        }],
        "time_column": "timestamp",
        "start_seconds": -30 * 86_400,
        "stop_seconds": 0,
        "distinct": True,
        "columns": ["order_id", "amount", "channel"],
    }],
    "sequences": [{
        "name": "recent_orders",
        "frame": "orders",
        "columns": ["order_id", "amount"],
        "order": "descending",
        "limit": 64,
    }],
    "aggregates": [{
        "name": "mean_channel_spend",
        "frame": "orders",
        "groups": [{
            "keys": ["channel"],
            "values": [{
                "name": "channel_spend",
                "operation": "sum",
                "column": "amount",
            }],
        }],
        "operation": "mean",
        "column": "channel_spend",
    }],
}
result = engine.query_temporal(
    {"users": root_ids},
    {"users": cutoff_epoch_seconds},
    json.dumps(plan),
    output_format="model",
)
```

Windows are half-open relative to each aligned cutoff:
`[cutoff + start_seconds, cutoff + stop_seconds)`. Either bound may be null;
bounds require `time_column`. Selected rows are sorted by timestamp and then
primary key before a sequence optionally reverses and truncates them.
`distinct` deduplicates destination dense IDs per root.

Available reductions are `count`, `count_distinct`, `sum`, `mean`, `min`,
`max`, `std_population`, `std_sample`, `var_population`, `var_sample`,
`recency`, and `mode`. `groups` are evaluated in list order, so the output of
one grouping stage is the input to the next. Empty `count`, `count_distinct`,
and `sum` are zero; other empty reductions are null. A non-empty all-null sum
is null. Sample variance/deviation requires two non-null values, and mode ties
use a stable type/value ordering.

The full row kernel still scans the reached adjacency; a sequence limit does
not stop traversal, and aggregates still consider the entire selected frame.
Neither API binary-prunes a time window within adjacency storage, applies
arbitrary row predicates, follows field-level joins, or executes across a
distributed mesh.
Those limitations are explicit so callers do not mistake this boundary for the
much broader training-table query engine.

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
