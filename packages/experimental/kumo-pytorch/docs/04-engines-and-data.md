# 4. Engines and data

A contract says what to build: the values each example may read at its
prediction time, and how they're encoded and arranged for the model. An engine
is what builds it. You give `prepare` a contract, a `Dataset` that says where
each table's bytes are, and an engine, and the engine reads the rows and
produces the batches. There's no default engine, so you always pass one.

kumo-pytorch ships two engines. The DiskGraph engine (`DiskGraphEngine`), from
[kumo-engine](https://github.com/NVIDIA/kumo-engine#readme), is the one to use
for real data. The in-memory engine (`InMemoryEngine`) runs the contract in
plain Python over a few rows held in memory. It's meant for trying the library
out on toy data you want to print and inspect, which is why most of this
guide's small examples use it. This page explains what an engine does, shows
how to bind your data, covers each of the two engines, and ends with how to
choose one and what happens when it can't run your contract.

## What an engine is

An engine is an implementation of the `Engine` interface. It checks your
contract and refuses anything it can't run, naming the parts involved and
suggesting a next step. Most refusals happen before any data is read. The DiskGraph
engine lists what it doesn't support yet in `DiskGraphEngine.unsupported`, and
the [limitations](limitations.md#diskgraph) page shows the same list.

The contract, the dataset and the engine change independently. Moving your
files changes the dataset, pointing the DiskGraph engine at a different index
directory changes the engine, and neither touches the contract. The same
contract runs on both engines that ship with the package. Because the engine is
an interface, other implementations are possible too, for example in a fork of
the repository, although we don't offer a stable API for writing them yet.

## Binding data

We'll use a small schema with users and their orders, and a contract that reads
each user's total spend and order history before the prediction time. The
example writes the rows to Parquet files in a temporary directory, the way your
own tables would already sit on disk, and binds each table to its file.

```python
import tempfile
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kumo.pytorch import api

schema = api.DataSchema(
    tables=(
        api.TableSpec(
            name="users",
            primary_key="user_id",
            columns={"user_id": api.DType(name="int64")},
            time_column=api.NON_TEMPORAL,
        ),
        api.TableSpec(
            name="orders",
            primary_key="order_id",
            columns={
                "order_id": api.DType(name="int64"),
                "user_id": api.DType(name="int64"),
                "occurred_at": api.DType(name="timestamp", unit="s", timezone="UTC"),
                "amount": api.DType(name="float64"),
            },
            time_column="occurred_at",
        ),
    ),
    relations=(
        api.RelationSpec(
            name="user_orders",
            left_table="users",
            left_key="user_id",
            right_table="orders",
            right_key="user_id",
        ),
    ),
)
history = api.Rows(path=("user_orders",), time=api.TimeWindow(column="occurred_at", stop=0))
contract = api.DataLoaderSpec(
    schema=schema,
    sample=api.SampleSpec(
        entity=api.EntitySpec(table="users", entity_type="user"),
        entity_values={"spend": api.Aggregate(source=history, operation="sum", column="amount")},
        events={
            "orders": api.EventSpec(rows=history, fields={"amount": api.Field(column="amount")})
        },
    ),
    batch=api.BatchSpec(
        inputs={
            "user": api.Input(
                source=api.EntitySource(),
                representation=api.Fields(fields=(api.Encode(field="spend", encoding=api.Raw()),)),
                layout=api.Separate(),
            ),
            "orders": api.Input(
                source=api.EventSource(events=("orders",)),
                representation=api.Fields(fields=(api.Encode(field="amount", encoding=api.Raw()),)),
                layout=api.Padded(),
            ),
        },
        batching=api.Batching(max_samples=1),
    ),
)

users = [{"user_id": 1}]
orders = [
    {"order_id": 10, "user_id": 1, "occurred_at": datetime(2024, 1, 2), "amount": 20.0},
    {"order_id": 11, "user_id": 1, "occurred_at": datetime(2024, 1, 7), "amount": 35.0},
    {"order_id": 12, "user_id": 1, "occurred_at": datetime(2024, 1, 12), "amount": 99.0},
]
root = Path(tempfile.mkdtemp())
pq.write_table(pa.Table.from_pylist(users), root / "users.parquet")
orders_schema = pa.schema(
    [
        ("order_id", pa.int64()),
        ("user_id", pa.int64()),
        ("occurred_at", pa.timestamp("s", tz="UTC")),
        ("amount", pa.float64()),
    ]
)
pq.write_table(pa.Table.from_pylist(orders, schema=orders_schema), root / "orders.parquet")
parquet = api.Dataset(
    tables={
        "users": api.Parquet(path=str(root / "users.parquet")),
        "orders": api.Parquet(path=str(root / "orders.parquet")),
    }
)
request = api.SampleRequest(
    sample_id="user-1-jan-10", entity_id=1, as_of_time=datetime(2024, 1, 10)
)
```

Each table in the schema is bound to one Parquet source, which can be a single
file, a directory or a glob. A dataset binds exactly the contract's tables, and
it doesn't name an engine. The DiskGraph engine reads `Parquet` sources, and the
in-memory engine reads `InMemoryRows`, which we'll come to below.

You don't have to repeat column types that the files already record. With
Parquet, you can omit `TableSpec.columns`, or give a column's type as `None`,
and the types are read from the file metadata. `contract.resolve(dataset=...)`
and `prepare` resolve them without reading any rows. Files used to infer a
column's type must agree on that type. The resolved contract is saved and
fingerprinted, and rebinding checks that the source types are compatible.

For a declared `int64` column, DiskGraph also accepts Parquet `int8`,
`int16`, `int32`, and scale-zero `decimal128` or `decimal256` columns.
The contract and batches use `int64`. Ingest fails if any decimal value
falls outside the signed 64-bit range.

## The DiskGraph engine

The DiskGraph engine is built on kumo-engine's DiskGraph, which the
[kumo-engine README](https://github.com/NVIDIA/kumo-engine#readme) describes.
[Installation](installation.md) explains how to install it.

### Building and reusing the index

The first time `prepare` runs, `DiskGraphEngine` builds an index from the
Parquet files in the dataset. By default it allocates a physical build ID and
writes the base under the configured Kumo scratch root at
`indexes/<index_build_id>`. Reuse the engine instance across preparations, or
consume a completed core snapshot with `from_snapshot(snapshot)`.

<!-- test: diskgraph -->
```python
import kumo.settings

kumo.settings.configure(scratch_dir=root)
engine = api.DiskGraphEngine(num_shards=1)
with (
    api.prepare(contract=contract, dataset=parquet, engine=engine) as plan,
    plan.materialize(anchors=(request,)) as task,
):
    batch = next(task.batches(output_format="model"))
print("spend:", batch.inputs["user"]["spend"]["values"].tolist())
print("amounts:", batch.inputs["orders"]["amount"]["values"].tolist())
```

<!-- test: output -->
```text
spend: [55.0]
amounts: [[20.0, 35.0]]
```

The January 12 order is after the January 10 prediction time, so it is not
read.

The index holds rows, key maps and relations. It holds no anchors, labels or
fitted encoders, so several contracts can share one base through the same engine
instance or snapshot binding. The managed layout is:

```text
<scratch_root>/indexes/<index_build_id>/
<scratch_root>/tasks/<index_build_id>/<task_id>/
<scratch_root>/cache/<index_build_id>/
```

The build ID identifies one physical ingest, independently of schema and data
version. Each fresh dataset engine allocates a different ID and ingests its own
base, even for identical data. There is no automatic reuse across engines by
graph or data version, so each adds ingestion time and disk usage. Reuse a core
`Graph.snapshot()` through `DiskGraphEngine.from_snapshot(snapshot)` to share
the producer's base across engines or processes. Kumo allocates IDs and chooses
index, task and cache paths internally; callers configure only the process-wide
scratch root.

Fitting reads any statistics already stored under the base's `col_stats/` and
writes missing statistics, tokenizers and token lengths under the managed
`cache/<index_build_id>/col_stats/`. Each encoder cutoff has a directory with a
`<table>.col_stats.json` file per table. A later fit at the same cutoff computes
only missing columns or statistics. Ingest computes statistics for the cutoffs
in `DiskGraphEngine(..., encoder_cutoffs=["2024-01-08"])`. A `Text` encoder's
tokenizer is trained on the first fit at its cutoff and saved beside them.

The manifest binds the schema, source files and resolved types. If it differs
from the dataset and schema you pass, `prepare` refuses reuse. The shard count
is not part of that check; an existing index keeps the shard count it was built
with. Use `DiskGraphEngine.from_snapshot(snapshot)` to prepare an existing index
without source ingestion.

The removed `ingest="never"` option guarded against unexpected ingestion: `prepare`
refused a missing base instead of ingesting it. Dataset engine construction no
longer exposes that guard; its first `prepare` ingests the dataset. Migrate
callers requiring an existing base to `DiskGraphEngine.from_snapshot(snapshot)`
and omit the Dataset from `prepare`. Snapshot binding validates the completed
base and refuses missing or incompatible state without rebuilding it or reading
the original source tables.

### Storage lifetime and cleanup

Closing a plan or task keeps `indexes/<index_build_id>/` and
`cache/<index_build_id>/`. Task cleanup removes its own task directory;
empty parent directories can remain. There is no automatic garbage collection
for unused bases, caches or abandoned builds, so repeated runs accumulate disk
usage.

For a managed build you no longer need, manually remove its matching index,
cache and leftover task directories only after all engines, tasks and processes
using it have stopped and no retained snapshot needs its base. Keep builds and
caches needed for later reuse.

### Open an existing snapshot

`DiskGraphEngine.from_snapshot(snapshot)` consumes an existing `kumo.engine.GraphSnapshot` under that same root. The loader opens `snapshot.index_path` in place and writes tasks and caches under the configured root at `tasks/<index_build_id>/` and `cache/<index_build_id>/`. Snapshots without a physical build ID cannot be adopted. Cross-root snapshot import is outside this interface.

<!-- test: skip (snapshot, sample_spec, batch_spec and anchors belong to the caller) -->
```python
engine = api.DiskGraphEngine.from_snapshot(snapshot)
contract = api.DataLoaderSpec(
    schema=engine.schema,
    sample=sample_spec,
    batch=batch_spec,
)
with api.prepare(contract=contract, engine=engine) as plan:
    with plan.materialize(anchors) as task:
        for batch in task.batches(output_format="columnar"):
            graph = batch.inputs["graph"]
```

`SampleSpec` and `BatchSpec` describe sampling and output encoding. `engine.schema` derives stored types, keys, clocks and relationships from native config and metadata. PyTorch-produced indexes expose logical relation aliases; core snapshots use native FK names. A compatible schema projection may rename those relations: binding checks endpoints, FK columns and direction. `prepare` rejects a Dataset alongside an existing-index engine. Anchors select entity/time requests; neighborhoods are sampled during iteration.

The base may be read-only. Task extensions and scalar-materialization scratch are temporary; closing a task removes them. Derived caches remain for reuse. The loader writes no new base metadata. Native extension metadata records the parent build ID, so opening a task against a different base is refused. Statistics may contain diagnostic paths to reused base files; those paths are not needed to restore fitted encoders. Concurrent requests may duplicate computation but do not discard published keys. Install DiskGraph from the matching kumo-engine revision for managed build and external cache support.

Opening requires the resolved `diskgraph_config.json` and the producer's completion metadata. Core snapshots use native inventory/integrity checks; PyTorch indexes use their manifest, published after ingest. Artifacts missing resolved config are refused. Native edge-table layouts, end-time validity and unsupported stored types (including lists) are refused explicitly.

For source ingestion and reuse, keep the same engine instance with the dataset.
The manifest checks source paths, file sizes and modification times; a changed
source or schema refuses reuse. To use a producer-managed snapshot after
sources move, open it with `from_snapshot(snapshot)` without a dataset.

`output_format="model"` lets the DiskGraph engine build the batch's tensors
column by column, without building a Python record per event. It refuses raw types
that a model tensor can't hold (strings, dates, decimals) before fetching any
rows, rather than converting them.

### What it supports

The DiskGraph engine supports entity fields, event histories, typed
calculations, aggregates, generated labels and recent or uniform graph neighborhoods. A
few contract features are outside what it runs today. `DiskGraphEngine.unsupported`
lists them, each with a code, what is refused and a next step, and the
[limitations](limitations.md#diskgraph) page shows the list. When a contract
uses one of them, the engine refuses it, in most cases at `check` or `prepare`
before reading any data (the list says when), as shown in [choosing an engine, and early
refusal](#choosing-an-engine-and-early-refusal).

### Anchor tables

An anchor is one example to build: an entity ID, a prediction time, and
optionally a split, labels, candidates and tracing metadata. For production
training, evaluation, offline scoring, and benchmarks, materialize a Parquet
`AnchorTable` with DiskGraph, then call `task.batches(split=..., output_format="model")`
without an explicit request argument. This selects native model iteration for
eligible contracts. Benchmarks of that iterator should verify the execution
route because some contracts still use Python assembly.

Use Python `SampleRequest` objects for tests, debugging, or small sets of
requests supplied at runtime. `task.batches(requests=...)` uses explicit
request batching, even on a task materialized from an `AnchorTable`. A task
materialized from a Python request sequence also stays request-backed when
you call `task.batches(split=...)`. DiskGraph can still sample the data for
these requests, but these calls bypass native AnchorTable batch iteration.

This example shows how to bind a Parquet anchor table and select its splits:

<!-- test: diskgraph -->
```python
anchor_rows = [
    {"sample_id": f"user-1-{day}", "user_id": 1, "as_of": datetime(2024, 1, day), "split": split}
    for day, split in ((5, "train"), (8, "train"), (10, "valid"), (13, "test"))
]
anchor_schema = pa.schema(
    [
        ("sample_id", pa.string()),
        ("user_id", pa.int64()),
        ("as_of", pa.timestamp("s", tz="UTC")),
        ("split", pa.string()),
    ]
)
pq.write_table(pa.Table.from_pylist(anchor_rows, schema=anchor_schema), root / "anchors.parquet")
anchors = api.AnchorTable(
    source=api.Parquet(path=str(root / "anchors.parquet")),
    entity_id="user_id",
    as_of_time="as_of",
    sample_id="sample_id",
    split="split",
)
with (
    api.prepare(contract=contract, dataset=parquet, engine=engine) as plan,
    plan.materialize(anchors=anchors, fit_split="train") as task,
):
    for split in ("train", "valid", "test"):
        spend = [b.inputs["user"]["spend"]["values"].tolist() for b in task.batches(split=split)]
        print(split, spend)
```

<!-- test: output -->
```text
train [[20.0], [55.0]]
valid [[55.0]]
test [[154.0]]
```

Each pass reads only the anchors in its own split. Encoders over computed values
fit on `fit_split`'s examples alone, so held-out anchors never shape them.
Encoders over stored columns instead fit on every table row created at or
before the encoder cutoff (by default the latest anchor time `fit_split`
selects), whichever split that row's entity belongs to; a non-temporal table
contributes every row. When anchors carry splits, `fit_split` is required.
It can name one split, a set of splits, or `api.ALL_ANCHORS`. Anchor tables can
also carry your own labels (`targets={"label": "column"}`), metadata columns
that are passed through to the batch, and nested candidate lists
(`api.CandidateColumn`).

For graph-neighborhood and sequence tasks, DiskGraph materializes only the anchors into a
task-owned native seed extension. The base graph stays unchanged. TT-NG then
owns bounded reads, deterministic epoch order and rank partitioning, and
produces topology and projected node features together. The existing fitted
encoders and batch layouts still apply. Original row positions supply omitted
sample IDs; repeated entity/time pairs remain separate examples. Native order
may differ from the in-memory engine, and padding remains explicitly marked.
Closing the task removes its temporary seed extension, not the base graph.

For eligible sequence contracts, the native model iterator reads histories and
merges event components before returning the batch to the existing encoders
and tensor layouts. Eligible length-aware contracts also plan batches and
reuse per-anchor lengths across epochs.

A selection spanning multiple split stages uses native stage batches. Each stage
is shuffled independently; batches never mix split stages. The output covers the
same anchors, while its order differs from the previous global Python shuffle.
Native graph output formats partition ranks, pad and drop partial
tails per stage. Refit does not combine partial tails from different splits.
With `drop_last=True`, either topology may drop a split completely when it is
smaller than one full batch. Padding uses a rank-owned anchor from another
selected stage if its current shard is empty; only a rank with no owned anchors
in the whole selection borrows marked padding from another rank. Padding-only
batches retain their source anchor's split and metadata.
Encoder fitting remains a separate stage and is not made streaming here.

The native model path avoids rebuilding a `SampleRequest` for each anchor.
Initial entity-key normalization still runs in Python, and nested list/struct
metadata uses per-row Arrow IPC serialization and decoding because RowPack has
no corresponding scalar. This is not a guarantee of zero per-row Python work.
Native chronological event merging requires timed components; an untimed merge
is rejected when history is attached, before the iterator consumes any batches.

Task extensions use the existing `train_tables` directory inside the graph index,
including on Windows. They require no symlinks. Closing a task releases its
combined reader before removing its task files.

Kumo-pytorch cutoffs round downward on every graph clock, preserving the
no-future-event boundary before 1970; stored datetimes round upward to their declared tick. Plain integer
anchors remain epoch seconds even on an ms/us/ns index. Use a datetime or typed
Arrow timestamp for an epoch-ms/us/ns anchor. Every frame in one temporal query
must agree on its cutoff unit (an omitted unit means seconds). Declared timestamp
fields retain their s/ms/us/ns precision through fitting, native model batches,
and artifact reload. Inspectable values finer than Python datetime's microsecond
resolution remain Arrow timestamp scalars; pandas is not required.

### Scalar features and labels in task preparation

For DiskGraph-backed training/BP, `Plan.materialize` computes declared scalar
features and generated labels once before fitting or iteration:

<!-- test: diskgraph -->
```python
with api.prepare(contract=contract, dataset=parquet, engine=engine) as plan:
    with plan.materialize(anchors, fit_split="train") as task:
        for batch in task.batches(split="train", output_format="model"):
            pass  # Supply these tensors to the model.
```

The materializer prepares the native aggregation plan once, reads projected
anchor chunks with the existing custom-TT reader, and writes rotating Parquet
parts under the managed task directory. It preserves anchor IDs, split,
metadata, and external labels alongside generated scalar features and labels.
The task owns these intermediate files and removes them on close.

Model-format iteration reads the prepared scalar columns directly; graph and
history retrieval remain dynamic. Explicit-request inspection and the unchanged
fitting lifecycle retain reference adapters. Fitting can still require
population-sized samples. Fixed anchor chunks do not bound the degree or group
cardinality of an individual anchor.

## The in-memory engine

`InMemoryEngine` runs the contract in plain Python over rows held in memory.
It's handy for trying out a contract on a few rows and printing what comes
back. It isn't built for real datasets, so use the DiskGraph engine for
those.

It reads `InMemoryRows` rather than Parquet. Python rows carry no type
metadata, so every column needs an explicit type in the schema.

```python
in_memory = api.Dataset(
    tables={"users": api.InMemoryRows(rows=users), "orders": api.InMemoryRows(rows=orders)}
)
# The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
with (
    api.prepare(contract=contract, dataset=in_memory, engine=api.InMemoryEngine()) as plan,
    plan.materialize(anchors=(request,)) as task,
):
    batch = next(task.batches())
print("spend:", batch.inputs["user"]["spend"]["values"].tolist())
print("amounts:", batch.inputs["orders"]["amount"]["values"].tolist())
```

<!-- test: output -->
```text
spend: [55.0]
amounts: [[20.0, 35.0]]
```

The batch is the same one the DiskGraph engine built from the Parquet files
above. Anchor tables work the same way on the in-memory engine, with
`api.InMemoryRows` as their source.

## Choosing an engine, and early refusal

Use the DiskGraph engine for real data, and the in-memory engine for tests and
toy examples. Most code constructs the engine directly, as the examples above
do. When the choice comes from a configuration file or an environment variable,
you can resolve a registered name to the same class instead:

```python
print("engines:", api.available_engines())
configured = api.engine_from_name("diskgraph")
print(type(configured).__name__)
```

<!-- test: output -->
```text
engines: ('diskgraph', 'in_memory')
DiskGraphEngine
```

Whichever engine you choose, it tells you at `prepare` if it can't run your
contract. `check(...)` returns the issues as a list, and `prepare(...)` raises
`ValidationError` with the same issues. Here we add a row filter to the history
behind an aggregate, which the DiskGraph engine doesn't support:

<!-- test: diskgraph -->
```python
from dataclasses import replace

large_orders = api.Rows(
    path=("user_orders",),
    time=api.TimeWindow(column="occurred_at", stop=0),
    filters=(api.RowFilter(column="amount", operation="greater_than", value=30.0),),
)
filtered = replace(
    contract,
    sample=replace(
        contract.sample,
        entity_values={
            "spend": api.Aggregate(source=large_orders, operation="sum", column="amount")
        },
    ),
)
for issue in api.check(contract=filtered, dataset=parquet, engine=engine):
    print(issue.path, issue.code)
    print("   ", issue.message)
    print("   ", issue.suggestion)
```

<!-- test: output -->
```text
sample.entity_values.spend.source.filters DG-ROW-FILTER
    DiskGraph does not support row filters
    precompute the condition as a column, or write only the matching rows to their own table, before writing the Parquet files
```

Each issue names the declaration it refers to, and carries a `message` saying
what is unsupported and a `suggestion` with a next step. A refusal from the
DiskGraph engine's list also carries the entry's `code`. The engine you pass
is always the one that builds your batches.

Next: [training batches](05-training-batches.md).
