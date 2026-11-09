# 5. Training batches

This page turns a contract into training batches: labels, fitting on the
training split, layouts, previews, deterministic order and rank sharding.

## A contract with a label

We'll use five users and their purchases. The label, which kumo-pytorch
computes for you, is how many purchases the user makes in the 7 days after the
prediction time.

```python
from kumo.pytorch import api

DAY = 86_400
users = [{"user_id": u, "segment": ("loyal", "casual")[u % 2]} for u in range(1, 6)]
purchases = [
    {"purchase_id": 100 * u + d, "user_id": u, "at": d * DAY + 3600, "amount": float(u + d)}
    for u in range(1, 6)
    for d in range(0, 30, u)
]
schema = api.DataSchema(
    tables=(
        api.TableSpec(
            name="users",
            primary_key="user_id",
            columns={"user_id": api.DType(name="int64"), "segment": api.DType(name="string")},
            time_column=api.NON_TEMPORAL,
        ),
        api.TableSpec(
            name="purchases",
            primary_key="purchase_id",
            columns={
                "purchase_id": api.DType(name="int64"),
                "user_id": api.DType(name="int64"),
                "at": api.DType(name="timestamp", unit="s", timezone="UTC"),
                "amount": api.DType(name="float64"),
            },
            time_column="at",
        ),
    ),
    relations=(
        api.RelationSpec(
            name="user_purchases",
            left_table="users",
            left_key="user_id",
            right_table="purchases",
            right_key="user_id",
        ),
    ),
)
history = api.Rows(path=("user_purchases",), time=api.TimeWindow(column="at", stop=0))
next_week = api.Rows(
    path=("user_purchases",), time=api.TimeWindow(column="at", start=0, stop=7 * DAY)
)
sample = api.SampleSpec(
    entity=api.EntitySpec(table="users", entity_type="user"),
    entity_values={"segment": api.Field(column="segment")},
    events={
        "purchases": api.EventSpec(
            rows=history, fields={"amount": api.Field(column="amount")}, limit=4, order="descending"
        )
    },
    targets={"next_week": api.Aggregate(source=next_week, operation="count")},
)
dataset = api.Dataset(
    tables={"users": api.InMemoryRows(rows=users), "purchases": api.InMemoryRows(rows=purchases)}
)
anchors = [
    api.SampleRequest(
        sample_id=f"user-{u}-day-{day}",
        entity_id=u,
        as_of_time=day * DAY,
        split="train" if day < 20 else "valid",
    )
    for u in range(1, 6)
    for day in (10, 15, 20)
]


def contract(layout: api.SequenceLayout) -> api.DataLoaderSpec:
    batch = api.BatchSpec(
        inputs={
            "user": api.Input(
                source=api.EntitySource(),
                representation=api.Fields(
                    fields=(api.Encode(field="segment", encoding=api.Vocabulary()),)
                ),
                layout=api.Separate(),
            ),
            "recent": api.Input(
                source=api.EventSource(events=("purchases",)),
                representation=api.Fields(
                    fields=(api.Encode(field="amount", encoding=api.ZScore()),)
                ),
                layout=layout,
            ),
        },
        targets={"active": api.LabelTarget(source="next_week", encoding=api.Raw())},
        batching=api.Batching(max_samples=4),
    )
    return api.DataLoaderSpec(schema=schema, sample=sample, batch=batch)
```

Generated labels live in `SampleSpec.targets` and must look strictly forward:
a target window starts at or after zero, and its start is excluded, so an event
at the prediction time is history, never label. `BatchSpec.targets` then
selects and encodes each label; declaring a label never puts it into an input.
Labels computed elsewhere are declared as `api.ExternalTarget(dtype=...)` and
supplied per request or per anchor-table column.

## Fit on the training split

```python
# The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
plan = api.prepare(contract=contract(api.Padded()), dataset=dataset, engine=api.InMemoryEngine())
task = plan.materialize(anchors=anchors, fit_split="train")
print("train/valid anchors:", task.count(split="train"), task.count(split="valid"))
batch = next(task.batches(split="train"))
print("active:", batch.targets["active"]["values"].tolist())
print("recent amounts:", tuple(batch.inputs["recent"]["amount"]["values"].shape))
```

<!-- test: output -->
```text
train/valid anchors: 10 5
active: [7, 7, 4, 3]
recent amounts: (4, 4)
```

`fit_split="train"` fits the segment vocabulary and the amount statistics on
training anchors only; every split is then served with that same frozen state.
A plan holds engine resources, so use it as a context manager (or call
`plan.close()`); keep it open while you use its tasks.

## Layouts

The same input can be stored padded (a rectangle plus a validity mask) or
packed (one flat buffer plus offsets). The sampled events are identical:

```python
padded = task.batches(split="valid")
# The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
with api.prepare(
    contract=contract(api.Packed()), dataset=dataset, engine=api.InMemoryEngine()
) as packed_plan:
    packed_task = packed_plan.materialize(anchors=anchors, fit_split="train")
    packed = next(packed_task.batches(split="valid"))
first = next(padded)
print("padded:", first.inputs["recent"]["valid"].sum(dim=1).tolist())
print("packed offsets:", packed.inputs["recent"]["offsets"].tolist())
```

<!-- test: output -->
```text
padded: [4, 4, 4, 4]
packed offsets: [0, 4, 8, 12, 16]
```

Grouped token histories use `api.Nested(fields=..., events=...)` to choose the
field and event axes independently. Packing keeps event and sample boundaries
as offsets. If you want one flat token stream instead, use
`history=api.Flatten(...)`.

## Preview

`preview` shows what a few anchors produce, raw and encoded, without mutating
fitted state. It is the quickest way to check a contract against your data:

```python
result = api.preview(task, anchors[:2])
print(len(result.samples), "samples;", tuple(result.batch.inputs))
```

<!-- test: output -->
```text
2 samples; ('user', 'recent')
```

## Deterministic order, epochs and ranks

`task.batches` takes `split` (or explicit `requests`), `batch_size`, `shuffle`,
`seed`, `epoch`, `rank` and `world_size`. The same arguments give the same
batches, so a run can be resumed or reproduced.

For multi-process training, ranks receive disjoint anchors. A short rank
repeats one of its own anchors to keep step counts equal, and marks the copy in
`metadata["is_padding"]`; exclude marked rows from losses and metrics:

```python
for rank in range(2):
    ids, padding = [], []
    for b in task.batches(split="valid", rank=rank, world_size=2, shuffle=True, seed=7, epoch=0):
        ids += b.metadata["sample_ids"]
        padding += b.metadata["is_padding"].tolist()
    print(rank, len(ids), "anchors,", sum(padding), "padding")
```

<!-- test: output -->
```text
0 3 anchors, 0 padding
1 3 anchors, 1 padding
```

`api.Batching` also offers token budgets (`max_tokens` with `length_input`),
length bucketing and `drop_last`. Token budgets count actual emitted
positions, padding included, and are refused with `world_size > 1`, where
ranks could otherwise take different numbers of steps.

## Inputs, targets and devices

A `Batch` keeps `inputs`, `targets` and `metadata` apart. `metadata` carries
sample and entity IDs, prediction times, splits and fingerprints. Don't put
model features in `metadata`; your model should read only `inputs`. `batch.to(device)` moves every nested tensor.
`output_format="model"` lets the DiskGraph engine build tensors directly (and
refuses raw types a tensor can't hold before fetching); the default inspectable
format keeps such values as Arrow arrays.

```python
print(sorted(first.metadata))
print(first.to("cpu").inputs["user"]["segment"]["values"].device)
```

<!-- test: output -->
```text
['artifact_fingerprint', 'as_of_time', 'contract_fingerprint', 'entity_ids', 'is_padding', 'request_metadata', 'sample_ids', 'splits']
cpu
```

## Self-supervised targets

Some targets are derived from an input rather than from future rows:

- `api.MaskedValuesTarget(input=...)` hides selected values of one input behind
  a reserved mask ID and asks the model to recover them (masked pre-training).
- `api.NextTokenTarget(input=...)` shifts an already-encoded flattened stream by
  one (causal pre-training).
- `api.AlignedEventTarget(input=..., fields=...)` predicts withheld fields of the
  events an input keeps.
- `api.MatchingGroupTarget(input=..., fields=...)` groups samples by raw values,
  for contrastive objectives.

[Model input shapes](07-model-input-shapes.md) shows each in a complete
batch, and the [fraud fine-tune example](../examples/pragma-fraud/)
trains with a masked-value target.

## Reusing fitted state

A second task can reuse fitted state instead of fitting again, for example to
score new anchors with the training vocabulary, or to fine-tune on a new label:

```python
scoring = plan.materialize(anchors=anchors[:3], fitted=task.fitted_state)
fine_tune = plan.materialize(anchors=anchors, fitted=task.fitted_state.inputs, fit_split="train")
print(scoring.fitted_state == task.fitted_state)
plan.close()
```

<!-- test: output -->
```text
True
```

The complete state requires the same contract; the input half,
`fitted_state.inputs`, works for any contract with the same inputs and lets its
targets fit.
[Inference and serving](06-inference-and-serving.md) saves this state with the
contract as an artifact.

## Training on the DiskGraph engine

The examples above run on the in-memory engine so they can print small results.
A real training job reads Parquet files through the DiskGraph engine, and the
contract doesn't change. Here we write the same users, purchases and anchors to
Parquet files, and fit on the training split as before. On the DiskGraph engine,
the anchors are a Parquet `AnchorTable` too.

<!-- test: diskgraph -->
```python
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

data = Path(tempfile.mkdtemp())
stamp = pa.timestamp("s", tz="UTC")
pq.write_table(pa.Table.from_pylist(users), data / "users.parquet")
pq.write_table(
    pa.Table.from_pylist(
        purchases,
        schema=pa.schema(
            [
                ("purchase_id", pa.int64()),
                ("user_id", pa.int64()),
                ("at", stamp),
                ("amount", pa.float64()),
            ]
        ),
    ),
    data / "purchases.parquet",
)
pq.write_table(
    pa.Table.from_pylist(
        [
            {
                "sample_id": a.sample_id,
                "user_id": a.entity_id,
                "as_of": datetime.fromtimestamp(a.as_of_time, UTC),
                "split": a.split,
            }
            for a in anchors
        ],
        schema=pa.schema(
            [
                ("sample_id", pa.string()),
                ("user_id", pa.int64()),
                ("as_of", stamp),
                ("split", pa.string()),
            ]
        ),
    ),
    data / "anchors.parquet",
)

parquet = api.Dataset(
    tables={
        "users": api.Parquet(path=str(data / "users.parquet")),
        "purchases": api.Parquet(path=str(data / "purchases.parquet")),
    }
)
anchor_table = api.AnchorTable(
    source=api.Parquet(path=str(data / "anchors.parquet")),
    entity_id="user_id",
    as_of_time="as_of",
    sample_id="sample_id",
    split="split",
)
engine = api.DiskGraphEngine(index=data / "index")
with (
    api.prepare(contract=contract(api.Padded()), dataset=parquet, engine=engine) as disk_plan,
    disk_plan.materialize(anchors=anchor_table, fit_split="train") as disk_task,
):
    print("train/valid anchors:", disk_task.count(split="train"), disk_task.count(split="valid"))
    batch = next(disk_task.batches(split="train", output_format="model"))
    print("active:", batch.targets["active"]["values"].tolist())
```

<!-- test: output -->
```text
train/valid anchors: 10 5
active: [7, 7, 4, 3]
```

These are the same anchors and labels as in the first example on this page.
`output_format="model"` lets the DiskGraph engine assemble each batch's tensors
column by column, without Python records. In a multi-process job, each process passes its own `rank` and
`world_size` to `batches`, as shown in [deterministic order, epochs and
ranks](#deterministic-order-epochs-and-ranks). See [the DiskGraph
engine](04-engines-and-data.md#the-diskgraph-engine) for how the index is built
and reused.

Next: [inference and serving](06-inference-and-serving.md).
