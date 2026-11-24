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
windows must end before the earliest query. Query targets never enter payloads.

FM derives schema metadata from the adopted engine. For a relation alias or
projection, pass `schema=api.DataSchema(...)`; PyTorch validates it against the
stored graph. Supported plans use one neighborhood, bounded recent fanouts,
scalar FM wire types, and `Aggregate(Rows(..., time=TimeWindow(start=0,
stop=positive)))` or a `Compare` over that aggregate. Multiple graph layers run
in one native sampling plan. Timeless destinations and uniform sampling fail
explicitly. Classification needs a two-entry raw-label-to-name map; regression
accepts a numeric aggregate. Labels and graphs use separate materializations
against the same base.

`loader.materialize(...).load()` and `loader.load(...)` return `PayloadBatch`.
Its Arrow tables and metadata survive request and loader closure. Row, column
and decoded-byte budgets apply during assembly; HTTP byte budgets apply during
encoding. Arrays encoding requires the built C++ extension, with no Python
fallback; Arrow IPC is an alternate wire format.

## Source installation

Build and install matching DiskGraph, settings and PyTorch wheels from this
workspace, then `uv build --wheel --package kumo-fm`. The [local SDM example](
examples/local_sdm/README.md) documents its separate optional inference setup.
