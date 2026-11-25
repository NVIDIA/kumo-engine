# Kumo FM payload loader

Experimental `kumo.fm` prepares retained Arrow payloads from completed engine
snapshots, using the public PyTorch sampling API. Arrays and Arrow IPC encode
the same retained tables without sampling again. SDM is needed only to run the
optional local model example.

## Snapshot workflow

Configure storage once with `kumo.settings.configure(scratch_dir=...)`. Build a
`kumo.engine.Graph` from connector tables and call `graph.snapshot()`, then
adopt that handle with `api.DiskGraphEngine.from_snapshot(snapshot)`.

Declare a public PyTorch `SampleSpec` with a future target and one neighborhood:

```python
from kumo import fm
from kumo.pytorch import api

engine = api.DiskGraphEngine.from_snapshot(snapshot)
with fm.prepare(engine=engine, sample=sample, target="next_orders",
                task="binary_classification", classes={False: "no", True: "yes"},
                features={"users": ["user_id"], "orders": ["order_time"]}) as loader:
    batch = loader.load(
        context=fm.Context(entities=range(32), times=["2025-06-01T00:00:00Z"]),
        query=fm.Query(entities=range(8), at="2025-11-01T00:00:00Z"))

context_tables = batch.tables("context")
arrays_body = batch.encode("arrays")
ipc_body = batch.encode("arrow_ipc")
```

See [the runnable example](examples/local_sdm/connector_to_prediction.py) for a
complete `Graph` and `SampleSpec`. It writes small PyArrow tables to Parquet and
uses `FileConnector`; tests follow the same approach with hand-authored expected
labels and selected rows.

The snapshot must belong to the configured root when adopted. The engine keeps
that root after settings change; tasks and caches use managed sibling paths and
the base stays unchanged. For an exported snapshot, configure the destination
root, call `GraphSnapshot.load_from_connector(...)`, then adopt its handle.
Task cleanup retains base indexes and caches; there is no automatic collection.

## Occurrences and supported plans

`Context(entities=..., times=...)` forms entities × times with time as the outer
axis. `Query(entities=..., at=...)` uses one time. Either accepts ordered
`pairs=[(entity, time[, occurrence_id]), ...]`. IDs default to zero-based
positions within each split; repeated pairs remain distinct. Context label
windows must end before the earliest query. Every context target must be non-null;
for example, `mean` over an empty future window produces a null label. Loading
rejects such requests and reports the target name and affected occurrence IDs
without dropping occurrences. `count` returns zero for an empty window.
Query targets never enter payloads.

Source primary keys must not be named `instance_id`, which FM reserves for
occurrence IDs. Source features named `instance_id` cannot be projected either.
Graph directions follow the declared relation endpoints: forward goes left to
right, and reverse goes right to left. The `max_instances` budget bounds each split before expanding
entities × times or validating explicit pairs.

FM derives payload types only for the selected graph tables and their requested
features, primary/foreign keys and clocks. Other source columns and unrelated
tables do not need FM-compatible types or keys. Requested `int32` and `float32`
features widen to `int64` and `float64`; target evaluation keeps the source schema.
For a relation alias or projection, pass `schema=api.DataSchema(...)`; PyTorch validates it against the
stored graph. Supported plans use one neighborhood, bounded recent fanouts,
scalar FM wire types, and `Aggregate(Rows(..., time=TimeWindow(start=0,
stop=positive)))` or a `Compare` over that aggregate. Multiple graph layers run
in one native sampling plan. Timeless destinations and uniform sampling fail
explicitly. `prepare` accepts only `task="regression"` or
`task="binary_classification"`; the ambiguous `"classification"` alias is refused.
Binary classification needs a two-entry raw-label-to-name map; regression accepts
a numeric aggregate. Labels and graphs use separate materializations
against the same base. Request anchors are passed in memory; any task resources
are managed by PyTorch under the engine's captured Kumo root.

`prepare(...)` is the loader factory. `loader.materialize(...)` validates and
retains bounded occurrences in a `PreparedRequest`; it does not sample or hold
native task resources. `request.load()` samples the occurrences and returns a
`PayloadBatch`; `loader.load(...)` combines those steps. Each load samples anew.
Its Arrow tables and metadata survive request and loader closure. Label alignment,
class mapping and related-row ordering use columnar kernels; labels remain Arrow
arrays through instance-table assembly. Row, column
and decoded-byte budgets apply during assembly; HTTP byte budgets apply during
encoding. Arrays encoding uses DiskGraph’s Rust extension with the GIL released; build
DiskGraph from the same checkout. FM ships a pure Python wheel. Arrow IPC is
an alternate wire format.

## Budgets

Pass a mapping such as `budgets={"max_instances": 500}` to `prepare` to override
the defaults below. Keys must match exactly; unknown keys and values other than
positive integers are refused. Booleans are not accepted as integers. Limits
are captured during preparation, independently of later changes to the mapping.

| Key | Default | Scope |
| --- | --- | --- |
| `max_instances` | 10,000 | Occurrences in each context/query split, checked before expansion |
| `max_rows_per_table` | 1,000,000 | Rows in each final instance/related table per split; also checked during internal assembly |
| `max_columns` | 500 | Columns in each payload table, including generated IDs and relationship columns |
| `max_candidate_rows` | 1,000,000 | Sampled node-plus-edge rows in each internal graph batch; this scope follows `batch_size` |
| `max_decoded_arrow_bytes` | 31,457,280 (30 MiB) | Sum of logical Arrow bytes across all retained tables and splits |
| `max_payload_bytes` | 31,457,280 (30 MiB) | Complete encoded JSON body, including metadata and arrays or base64 IPC |

## Wire output

`PayloadBatch.encode("arrays" | "arrow_ipc")` produces an experimental request
envelope with `model`, `task`, `schema` and `output` metadata. This package is a
producer of that envelope; it supplies no HTTP client or response consumer.
Sampled public columnar graphs feed FM tables directly. The loader caches its
wire schema, and each payload normalizes its retained tables once before either
encoding.
Tests compare arrays/IPC serialization with retained Arrow tables locally.
Those checks do not establish parity with a live HTTP endpoint; no live endpoint
or in-repository HTTP consumer is supplied or exercised by this PR.

## Source installation

Build and install matching DiskGraph, settings and PyTorch wheels from this
workspace, then `uv build --wheel --package kumo-fm`. The [local SDM example](
examples/local_sdm/README.md) documents its separate optional inference setup.

## Correctness tests

See [the testing guide](TESTING.md) for the fast CI suite, opt-in developer
campaigns, and regression coverage mapped to PR #391 review findings.
