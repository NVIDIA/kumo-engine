# 2. Contracts, step by step

A contract is three declarations: `DataSchema` (what is stored), `SampleSpec`
(what one example may read at its prediction time) and `BatchSpec` (what the
model receives). This page builds one contract over a small shop, then extends
it with a graph neighborhood and caller-supplied candidates. Every block runs
in order; paste them into one Python session.

These snippets use Python requests to inspect a few examples. For training,
evaluation, or benchmarking on DiskGraph, use the
[Parquet AnchorTable recipe](05-training-batches.md#training-on-the-diskgraph-engine).

## The schema: types, keys and clocks

Declare each stored column once with its raw scalar type, the relations between
tables, and a clock per table. Entity tables require a primary key. Related
event or transaction tables may set `primary_key=None`; each source row remains
a separate occurrence, including identical rows:

```python
import kumo.pytorch as kp

DAY = 86_400
timestamp = kp.DType(name="timestamp", unit="s", timezone="UTC")

schema = kp.DataSchema(
    tables=(
        kp.TableSpec(
            name="users",
            primary_key="user_id",
            columns={"user_id": kp.DType(name="int64"), "country": kp.DType(name="string")},
            time_column=kp.NON_TEMPORAL,
        ),
        kp.TableSpec(
            name="items",
            primary_key="item_id",
            columns={"item_id": kp.DType(name="int64"), "category": kp.DType(name="string")},
            time_column=kp.NON_TEMPORAL,
        ),
        kp.TableSpec(
            name="purchases",
            primary_key="purchase_id",
            columns={
                "purchase_id": kp.DType(name="int64"),
                "user_id": kp.DType(name="int64"),
                "item_id": kp.DType(name="int64"),
                "at": timestamp,
                "amount": kp.DType(name="float64"),
            },
            time_column="at",
        ),
    ),
    relations=(
        kp.RelationSpec(
            name="user_purchases",
            left_table="users",
            left_key="user_id",
            right_table="purchases",
            right_key="user_id",
        ),
        kp.RelationSpec(
            name="item_purchases",
            left_table="items",
            left_key="item_id",
            right_table="purchases",
            right_key="item_id",
        ),
    ),
)
```

- A dtype describes the stored value, not how the model sees it. An `int64` can
  later be returned raw, standardized, bucketed or treated as a category.
- Every table names its clock (`time_column`) or says `NON_TEMPORAL`. A
  timestamp column does not become a clock by itself.
- A relation is an equality join and can be followed in either direction.
- For Parquet, `columns` can be omitted (or a column given as `None`) and the
  types are read from file metadata; in-memory rows need explicit types.

Here's the data as Python rows. Times are in seconds, and we'll predict as of
day 10:

```python
rows = {
    "users": [{"user_id": 1, "country": "US"}, {"user_id": 2, "country": "GB"}],
    "items": [
        {"item_id": 10, "category": "books"},
        {"item_id": 11, "category": "games"},
        {"item_id": 12, "category": "music"},
    ],
    "purchases": [
        {"purchase_id": 1, "user_id": 1, "item_id": 10, "at": 2 * DAY, "amount": 12.0},
        {"purchase_id": 2, "user_id": 1, "item_id": 11, "at": 7 * DAY, "amount": 30.0},
        {"purchase_id": 3, "user_id": 2, "item_id": 10, "at": 8 * DAY, "amount": 18.0},
        {"purchase_id": 4, "user_id": 1, "item_id": 12, "at": 12 * DAY, "amount": 99.0},
    ],
}
dataset = kp.Dataset(tables={name: kp.InMemoryRows(rows=values) for name, values in rows.items()})
```

## The sample: what one example may know

An example is an entity (here a user) at a prediction time. The caller supplies
both in a `SampleRequest`; the `SampleSpec` says what may be read for it.

```python
past = kp.Rows(path=("user_purchases",), time=kp.TimeWindow(column="at", start=-30 * DAY, stop=0))

sample = kp.SampleSpec(
    entity=kp.EntitySpec(table="users", entity_type="user"),
    entity_values={
        "country": kp.Field(column="country"),
        "spend_30d": kp.Aggregate(source=past, operation="sum", column="amount"),
        "orders_30d": kp.Aggregate(source=past, operation="count"),
        "average_order": kp.Derived(operation="divide", inputs=("spend_30d", "orders_30d")),
    },
    events={
        "purchases": kp.EventSpec(
            rows=past,
            fields={
                "amount": kp.Field(column="amount"),
                "item_id": kp.Field(column="item_id"),
                "age_days": kp.EventTimeDelta(unit="days", dtype=kp.DType(name="float64")),
                "at": kp.Field(column="at"),
                "weekday": kp.Derived(
                    operation=kp.CalendarPart(part="weekday", timezone="UTC"),
                    inputs=("at",),
                ),
            },
            limit=20,
            order="descending",
        )
    },
)

requests = (
    kp.SampleRequest(sample_id="user-1", entity_id=1, as_of_time=10 * DAY),
    kp.SampleRequest(sample_id="user-2", entity_id=2, as_of_time=10 * DAY),
)
```

A few things to notice:

- Windows are half-open, `(start, stop]`, in seconds from the prediction time.
  `stop=0` includes rows exactly at the prediction time, so the purchase on
  day 12 is never visible to a day-10 example. Without a window, a table with a
  clock contributes every row at or before the prediction time.
- Aggregates (`sum`, `count`, `mean`, `min`, `max`, `mode`, `recency`, variance
  and standard deviation), arithmetic (`Derived`), calendar parts, value
  mappings (`MapValues`) and event ages (`EventTimeDelta`) are declared in the
  sample, not in an encoder, and their result types are inferred and checked.
- Timestamps, ages and calendar parts reach the model only if a field asks for
  them.
- Named declarations can be written in any order. Operand tuples, field tuples
  and relation paths keep their order.

## The batch: what the model receives

A `BatchSpec` holds named inputs. Each input picks a source from the sample,
an ordered tuple of fields with an explicit encoder each, and a layout:

```python
batch = kp.BatchSpec(
    inputs={
        "user": kp.Input(
            source=kp.EntitySource(),
            representation=kp.Fields(
                fields=(
                    kp.Encode(field="country", encoding=kp.Vocabulary()),
                    kp.Encode(field="spend_30d", encoding=kp.Raw()),
                    kp.Encode(field="average_order", encoding=kp.Raw()),
                )
            ),
            layout=kp.Separate(),
        ),
        "history": kp.Input(
            source=kp.EventSource(events=("purchases",)),
            representation=kp.Fields(
                fields=(
                    kp.Encode(field="amount", encoding=kp.ZScore()),
                    kp.Encode(field="item_id", encoding=kp.Vocabulary()),
                    kp.Encode(field="age_days", encoding=kp.Raw()),
                    kp.Encode(field="weekday", encoding=kp.Vocabulary()),
                ),
                row=kp.Columns(),
                history=kp.Grouped(),
            ),
            layout=kp.Padded(),
        ),
    },
    batching=kp.Batching(max_samples=2),
)
contract = kp.DataLoaderSpec(schema=schema, sample=sample, batch=batch)

# The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
with (
    kp.prepare(contract=contract, dataset=dataset, engine=kp.InMemoryEngine()) as plan,
    plan.materialize(anchors=requests) as task,
):
    result = next(task.batches(requests=requests))

user = result.inputs["user"]
history = result.inputs["history"]
print("spend_30d:", user["spend_30d"]["values"].tolist())
print("average_order:", user["average_order"]["values"].tolist())
print("age_days:", history["age_days"]["values"].tolist())
print("valid:", history["valid"].tolist())
```

<!-- test: output -->
```text
spend_30d: [42.0, 18.0]
average_order: [21.0, 18.0]
age_days: [[8.0, 3.0], [2.0, 0.0]]
valid: [[True, True], [True, False]]
```

User 1's day-12 purchase is excluded. `limit=20, order="descending"` keeps the
newest 20 events; the input delivers them oldest first unless
`EventSource(order=...)` says otherwise. User 2's single purchase is padded,
with `valid` marking the padding slot.

An input's representation makes three independent choices:

| Choice | Options |
| --- | --- |
| Row arrangement | `Columns()` keeps one typed array per field; `Tokens()` concatenates ID-producing fields into one token row; `Composite()` maps an ordered tuple of fields to one ID |
| History arrangement | `Grouped()` keeps one row per event; `Flatten(...)` concatenates a sample's events into one sequence with declared markers |
| Layout | `Separate()` for entity columns, `Padded()` or `Packed()` for sequences, `Nested(...)` for grouped token histories, `None` for graphs |

A packed layout still keeps every sequence boundary, as offsets. [Encodings and IDs](03-encodings-and-ids.md) covers encoders,
and [training batches](05-training-batches.md) covers layouts and targets.

## Graph neighborhoods

A neighborhood is another way to read the same relations: rows are nodes and
declared key relationships are typed edges. It belongs in the sample; the batch
only chooses node fields and encoders.

```python
neighborhood = kp.NeighborhoodSpec(
    layers=(
        kp.GraphLayer(
            samples=(
                kp.GraphSample(
                    relation="user_purchases", fanout=5, direction="forward", strategy="recent"
                ),
            )
        ),
        kp.GraphLayer(
            samples=(
                kp.GraphSample(
                    relation="item_purchases",
                    fanout=1,
                    direction="reverse",
                    strategy="uniform",
                ),
            )
        ),
    )
)
graph_sample = kp.SampleSpec(
    entity=kp.EntitySpec(table="users", entity_type="user"),
    neighborhoods={"bought": neighborhood},
)
graph_batch = kp.BatchSpec(
    inputs={
        "graph": kp.Input(
            source=kp.NeighborhoodSource(name="bought"),
            representation=kp.Graph(
                node_fields={
                    "items": kp.Fields(
                        fields=(kp.Encode(field="category", encoding=kp.Vocabulary()),)
                    ),
                    "purchases": kp.Fields(
                        fields=(kp.Encode(field="amount", encoding=kp.ZScore()),)
                    ),
                }
            ),
            layout=None,
        )
    },
    batching=kp.Batching(max_samples=2),
)
graph_contract = kp.DataLoaderSpec(schema=schema, sample=graph_sample, batch=graph_batch)

# The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
with (
    kp.prepare(contract=graph_contract, dataset=dataset, engine=kp.InMemoryEngine()) as plan,
    plan.materialize(anchors=requests) as task,
):
    graph = next(task.batches(requests=requests)).inputs["graph"]

print("node types:", tuple(sorted(graph["nodes"])))
print("item ids:", graph["nodes"]["items"]["ids"].tolist())
print("owning sample:", graph["nodes"]["items"]["batch"].tolist())
print("edge types:", tuple(sorted(graph["edges"])))
```

<!-- test: output -->
```text
node types: ('items', 'purchases', 'users')
item ids: [11, 10, 10]
owning sample: [0, 0, 1]
edge types: ('item_purchases', 'user_purchases')
```

- An omitted strategy resolves to `recent` for a destination with a declared
  clock and `uniform` for a clockless destination. Explicit `recent` requires
  a declared clock and keeps the newest neighbors at or before the anchor.
- Explicit `uniform` samples without replacement from all related rows,
  ignoring the anchor even if the destination has a clock. DiskGraph uses
  `DiskGraphEngine(random_seed=0)` by default; a fixed engine seed replays the
  same selections on the same index across epochs.
- Each example's graph is disjoint: item 10 appears once for each user that
  reached it. Edges keep the relation's declared left-to-right orientation even
  when traversed in reverse.

## Caller-supplied candidates

For ranking, the caller chooses which items to score; a `CandidateSetSpec`
says what to read for each candidate at the request's prediction time, and
labels align with the caller's order:

```python
candidate_sample = kp.SampleSpec(
    entity=kp.EntitySpec(table="users", entity_type="user"),
    candidate_sets={
        "items": kp.CandidateSetSpec(
            entity=kp.EntitySpec(table="items", entity_type="item"),
            values={
                "category": kp.Field(column="category"),
                "prior_buyers": kp.Aggregate(
                    source=kp.Rows(
                        path=("item_purchases",), time=kp.TimeWindow(column="at", stop=0)
                    ),
                    operation="count",
                ),
            },
            targets={"bought": kp.ExternalTarget(dtype=kp.DType(name="bool"))},
        )
    },
)
candidate_batch = kp.BatchSpec(
    inputs={
        "candidates": kp.Input(
            source=kp.CandidateSource(name="items"),
            representation=kp.Fields(
                fields=(
                    kp.Encode(field="category", encoding=kp.Vocabulary()),
                    kp.Encode(field="prior_buyers", encoding=kp.Raw()),
                )
            ),
            layout=kp.Padded(),
        )
    },
    targets={"bought": kp.CandidateTarget(input="candidates", source="bought", encoding=kp.Raw())},
    batching=kp.Batching(max_samples=1),
)
candidate_request = kp.SampleRequest(
    sample_id="user-1",
    entity_id=1,
    as_of_time=10 * DAY,
    candidates={
        "items": tuple(
            kp.CandidateRequest(
                entity_id=item, targets=(kp.TargetValue(name="bought", value=item == 12),)
            )
            for item in (10, 11, 12)
        )
    },
)
# The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
with (
    kp.prepare(
        contract=kp.DataLoaderSpec(schema=schema, sample=candidate_sample, batch=candidate_batch),
        dataset=dataset,
        engine=kp.InMemoryEngine(),
    ) as plan,
    plan.materialize(anchors=(candidate_request,)) as task,
):
    ranked = next(task.batches())

print("prior_buyers:", ranked.inputs["candidates"]["prior_buyers"]["values"].tolist())
print("bought:", ranked.targets["bought"]["values"].tolist())
print("candidate ids:", ranked.metadata["candidate_entity_ids"]["items"])
```

<!-- test: output -->
```text
prior_buyers: [[2, 1, 0]]
bought: [[False, False, True]]
candidate ids: ((10, 11, 12),)
```

Item 12's only purchase is on day 12, so at day 10 it has no prior buyers.
The DiskGraph engine does not run candidate sets yet (see
[limitations](limitations.md)).

## Authoring with Python objects (experimental)

`kumo.pytorch.build` lets you refer to tables and columns as Python objects
instead of strings, and produces the same `DataLoaderSpec`:

```python
from kumo.pytorch import build

users = build.Table(
    name="users",
    key="id",
    columns={"id": kp.DType(name="int64"), "age": kp.DType(name="int64")},
    time=kp.NON_TEMPORAL,
)
built = build.contract(
    schema=build.Schema(tables=(users,)),
    sample=build.Sample(entity=users, values={"age": build.Read(column=users.age)}),
    batch=build.Batch(
        inputs={
            "profile": build.Input(
                source=users,
                representation=kp.Fields(fields=(kp.Encode(field="age", encoding=kp.Raw()),)),
                layout=kp.Separate(),
            )
        }
    ),
)
print(type(built).__name__)
```

<!-- test: output -->
```text
DataLoaderSpec
```

See [authoring](api/authoring.md) in the API reference for every builder.

Next: [encodings and IDs](03-encodings-and-ids.md).
