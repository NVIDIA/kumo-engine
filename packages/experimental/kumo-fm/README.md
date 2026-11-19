# Kumo FM payload loader

The primary workflow starts with a graph built through `kumo.engine` connectors.
Adopt its snapshot with `api.DiskGraphEngine.from_snapshot(snapshot,
work_dir=...)`, declare a public PyTorch `SampleSpec` with one
`NeighborhoodSpec` and a bounded future `Aggregate` or `Compare` target, then
call `fm.prepare(engine=..., sample=..., target=..., task=..., features=...)`.
See the runnable
[`connector_to_prediction.py`](examples/local_sdm/connector_to_prediction.py).
FM derives table and relation metadata from `engine.schema`; it creates
context/query anchors, computes context labels in a separate public PyTorch
plan, samples one columnar graph plan over the same index, and aligns both by
occurrence ID. The base index remains unchanged. Columnar graph output does not
currently fuse computed labels into the graph batch.
For an advanced public relation alias or table projection, pass
`schema=api.DataSchema(...)`; PyTorch checks it against the adopted index.

`fm.Context(entities=..., times=...)` produces entities × times, with time as
the outer axis. `fm.Query(entities=..., at=...)` uses one query time. Either
accepts ordered `pairs=[(entity, time), ...]`; a third element supplies an
explicit int64 occurrence ID. Repeated pairs remain separate. Without explicit
IDs each split uses zero-based positions. Context's future label window must be
fully observable before the earliest query time. Query targets never enter the
request. `loader.materialize(context=..., query=...).load()` and the shortcut
`loader.load(context=..., query=...)` return the same `PayloadBatch` type as
the frozen-bundle path. `batch.tables(split)` and both `batch.encode(...)`
methods reuse its retained Arrow tables without sampling again.

The snapshot path currently accepts a single public neighborhood, bounded
`recent` fanouts, supported scalar FM wire types and `Aggregate(Rows(...,
time=TimeWindow(start=0, stop=positive)))` or `Compare` over that aggregate.
Multiple `GraphLayer`s in that neighborhood are passed to one native sampling
operation. Classification requires an explicit two-entry raw-label-to-name
`classes` map; regression accepts a numeric aggregate. Unsupported types,
timeless destinations and uniform sampling fail explicitly. Native labels and
graphs remain separate materializations over the adopted snapshot.

## Frozen bundle compatibility

Experimental `kumo-fm` builds the existing Kumo Relational HTTP request from
public `kumo.pytorch` columnar graph batches backed by DiskGraph. Sampling and
FM assembly happen once per `PreparedRequest.load()`. The returned
`PayloadBatch` retains typed Arrow tables; calls to `encode("arrays")` and
`encode("arrow_ipc")` serialize those same tables without sampling.

Build and install `kumo-pytorch` and DiskGraph from the same engine revision,
then build and install this package's wheel. The example under
`examples/local_sdm` additionally needs SDM and a compatible pretrained
checkpoint; SDM is not a runtime dependency.

```bash
uv build --wheel --package kumo-pytorch --out-dir dist
uv build --sdist --package kumo-fm --out-dir dist
uv build --wheel dist/kumo_fm-*.tar.gz --out-dir dist
uv pip install --python /path/to/venv/bin/python --torch-backend cpu \
  /path/to/matching/diskgraph-*.whl dist/kumo_pytorch-*.whl dist/kumo_fm-*.whl
```

```python
from kumo import fm

with fm.prepare("/data/bundle", "/data/workload.json", index="/data/fm-index") as loader:
    with loader.materialize(context=[101, 102], predict=[201]) as request:
        batch = request.load()
        context = batch.tables("context")
        print(context["instance_table"].schema)
        arrays_body = batch.encode("arrays")
        ipc_body = batch.encode("arrow_ipc")
```

The optional `context` and `predict` lists contain ordered instance IDs
already validated by the workload. Omit a list to use that split's frozen
instances. Pass `[]` to omit a split. A direct SDM prediction needs both.
`materialize()` prepares the requested anchors; `load()` samples and assembles
them. Each explicit `load()` can produce a new batch. Retained batches own
their Arrow references after request and loader closure.

`batch.metadata` and `batch.as_arrow_request()` provide defensive metadata
copies. `batch.tables(split)` returns the instance Arrow table and named
related Arrow tables. `batch.logical_arrow_bytes` sums their `Table.nbytes`;
it is neither peak process memory nor the encoded HTTP body size. Row and
column limits and the decoded Arrow limit apply during assembly. The final
JSON body limit applies only during encoding. Encoding failures leave the
batch usable.

The wire envelope, composite `(instance_id, primary_key)` identity, selected
`__fm_edge_*` relationship columns, large integer representation, and
context-only target follow the prior loader contract. Arrays remain the
default wire format; Arrow IPC is opt-in. No SDM conversion runs in this wheel.

## Compatibility

The production path accepts frozen `native_multihop` workloads whose graph
sampling can be expressed exactly as public `kumo.pytorch` recent sampling.
Uniform sampling into timeless tables and empty graph layers fail explicitly.
The old `local_native` and `temporal_query` selection implementations are
comparison references, not fallbacks in this package. A supplied index must
match the dataset snapshot, contract fingerprint, and recorded dependency
provenance; an unrelated or incomplete index is refused.

| Contract area | Status |
| --- | --- |
| One or both frozen context/predict splits; ordered instance selection | Supported |
| Classification and regression targets; context label availability checks | Supported |
| Temporal recent sampling, forward/reverse steps, multiple relations and hops | Supported through the public columnar graph API |
| Per-instance identity, selected edges, nulls, empty related tables, exact int64 IDs | Preserved in typed Arrow tables and both encodings |
| Bundle hashes, source schema, keys/FKs, clock precision, rows/columns/bytes | Checked before or during sampling/encoding |
| Uniform sampling into timeless destinations, empty graph layers | Explicitly rejected |
| `local_native` and `temporal_query` selection | Comparison-only; no production fallback |

| Prior entry point | This package |
| --- | --- |
| `Bundle(root)`, `Workload(bundle, spec)` | `fm.prepare(root, spec, index=...)` |
| `KumoPytorchBackend.prepare_batch(seeds)` | `loader.materialize(context=..., predict=...)` |
| `build_arrow_tables(backend, workload)` | `request.load().as_arrow_request()` |
| `build_payload(..., encoding="arrays")` | `batch.encode("arrays")` |
| `build_payload(..., encoding="arrow_ipc")` | `batch.encode("arrow_ipc")` |
| `arrow_to_sdm(request)` | `examples/local_sdm/adapter.py:arrow_to_sdm(batch)` |

Source baselines: fm-dataloader `b4516394e98929c5b4752745866d5e3138276255`;
engine `9637e9cc9e2da877959b3c6b9db88c0fc7e35f3f`. The public columnar
API landed in engine PRs #340 and #365.
