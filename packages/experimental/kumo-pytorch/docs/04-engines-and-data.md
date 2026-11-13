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
and `prepare` resolve them without reading any rows. The resolved contract is
what gets saved and fingerprinted, so if a column's type in the files later
changes, loading fails loudly instead of quietly reading a different type.

## The DiskGraph engine

The DiskGraph engine is built on kumo-engine's DiskGraph, which the
[kumo-engine README](https://github.com/NVIDIA/kumo-engine#readme) describes.
[Installation](installation.md) explains how to install it.

### Building and reusing the index

The first time `prepare` runs, `DiskGraphEngine` builds an index from the
Parquet files in the directory you give it. Later runs find the index there and
reuse it.

<!-- test: diskgraph -->
```python
engine = api.DiskGraphEngine(index=root / "index", ingest="if_missing", num_shards=1)
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
fitted encoders, so several contracts over the same tables can share one index.
Fitting keeps the column statistics of encoders over stored columns in
`col_stats/`, so it needs the index directory to be writable. Each encoder
cutoff has one directory there, with a `<table>.col_stats.json` file per table.
A later fit at the same cutoff computes only the columns and statistics those
files lack, and adds them. A `Text` encoder's tokenizer is kept beside them as
`<table>.<column>.text-<settings>.tokenizer.json`. Ingest computes them for the cutoffs in
`DiskGraphEngine(..., encoder_cutoffs=["2024-01-08"])`.
Its manifest records the schema, the files and their resolved types. If the
manifest doesn't match the dataset and schema you pass, `prepare` refuses the
index rather than reusing it, and it never overwrites it. The shard count isn't
part of that check: an index opened with a different `num_shards` is reused as
it was built. With
`ingest="never"`, `prepare` requires an index that already exists, which is
what you usually want in a scoring job.

`output_format="model"` lets the DiskGraph engine build the batch's tensors
column by column, without building a Python record per event. It refuses raw types
that a model tensor can't hold (strings, dates, decimals) before fetching any
rows, rather than converting them.

### What it supports

The DiskGraph engine supports entity fields, event histories, typed
calculations, aggregates, generated labels and recent graph neighborhoods. A
few contract features are outside what it runs today. `DiskGraphEngine.unsupported`
lists them, each with a code, what is refused and a next step, and the
[limitations](limitations.md#diskgraph) page shows the list. When a contract
uses one of them, the engine refuses it, in most cases at `check` or `prepare`
before reading any data (the list says when), as shown in [choosing an engine, and early
refusal](#choosing-an-engine-and-early-refusal).

### Anchor tables

An anchor is one example to build: an entity ID, a prediction time, and
optionally a split, labels, candidates and tracing metadata. A few
`SampleRequest`s are enough for a quick check. For real work, you write the
anchors as a table and bind it with an `AnchorTable`. On the DiskGraph engine,
the anchor table is a Parquet file:

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
configured = api.engine_from_name("diskgraph", index=root / "index", ingest="never")
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
