# 2. Contracts, step by step

A contract is three declarations: `DataSchema` (what is stored), `SampleSpec`
(what one example may read at its prediction time) and `BatchSpec` (what the
model receives). This page builds one contract over a small shop, then extends
it with a graph neighborhood and caller-supplied candidates. Every block runs
in order; paste them into one Python session.

## The schema: types, keys and clocks

Declare each stored column once with its raw scalar type, a primary key per
table, the relations between tables, and a clock per table:

```python
from kumo.pytorch import api

DAY = 86_400
timestamp = api.DType(name="timestamp", unit="s", timezone="UTC")

schema = api.DataSchema(
    tables=(
        api.TableSpec(
            name="users",
            primary_key="user_id",
            columns={"user_id": api.DType(name="int64"), "country": api.DType(name="string")},
            time_column=api.NON_TEMPORAL,
        ),
        api.TableSpec(
            name="items",
            primary_key="item_id",
            columns={"item_id": api.DType(name="int64"), "category": api.DType(name="string")},
            time_column=api.NON_TEMPORAL,
        ),
        api.TableSpec(
            name="purchases",
            primary_key="purchase_id",
            columns={
                "purchase_id": api.DType(name="int64"),
                "user_id": api.DType(name="int64"),
                "item_id": api.DType(name="int64"),
                "at": timestamp,
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
        api.RelationSpec(
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
dataset = api.Dataset(tables={name: api.InMemoryRows(rows=values) for name, values in rows.items()})
```

## The sample: what one example may know

An example is an entity (here a user) at a prediction time. The caller supplies
both in a `SampleRequest`; the `SampleSpec` says what may be read for it.

```python
past = api.Rows(path=("user_purchases",), time=api.TimeWindow(column="at", start=-30 * DAY, stop=0))

sample = api.SampleSpec(
    entity=api.EntitySpec(table="users", entity_type="user"),
    entity_values={
        "country": api.Field(column="country"),
        "spend_30d": api.Aggregate(source=past, operation="sum", column="amount"),
        "orders_30d": api.Aggregate(source=past, operation="count"),
        "average_order": api.Derived(operation="divide", inputs=("spend_30d", "orders_30d")),
    },
    events={
        "purchases": api.EventSpec(
            rows=past,
            fields={
                "amount": api.Field(column="amount"),
                "item_id": api.Field(column="item_id"),
                "age_days": api.EventTimeDelta(unit="days", dtype=api.DType(name="float64")),
                "at": api.Field(column="at"),
                "weekday": api.Derived(
                    operation=api.CalendarPart(part="weekday", timezone="UTC"),
                    inputs=("at",),
                ),
            },
            limit=20,
            order="descending",
        )
    },
)

requests = (
    api.SampleRequest(sample_id="user-1", entity_id=1, as_of_time=10 * DAY),
    api.SampleRequest(sample_id="user-2", entity_id=2, as_of_time=10 * DAY),
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
batch = api.BatchSpec(
    inputs={
        "user": api.Input(
            source=api.EntitySource(),
            representation=api.Fields(
                fields=(
                    api.Encode(field="country", encoding=api.Vocabulary()),
                    api.Encode(field="spend_30d", encoding=api.Raw()),
                    api.Encode(field="average_order", encoding=api.Raw()),
                )
            ),
            layout=api.Separate(),
        ),
        "history": api.Input(
            source=api.EventSource(events=("purchases",)),
            representation=api.Fields(
                fields=(
                    api.Encode(field="amount", encoding=api.ZScore()),
                    api.Encode(field="item_id", encoding=api.Vocabulary()),
                    api.Encode(field="age_days", encoding=api.Raw()),
                    api.Encode(field="weekday", encoding=api.Vocabulary()),
                ),
                row=api.Columns(),
                history=api.Grouped(),
            ),
            layout=api.Padded(),
        ),
    },
    batching=api.Batching(max_samples=2),
)
contract = api.DataLoaderSpec(schema=schema, sample=sample, batch=batch)

# The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
with (
    api.prepare(contract=contract, dataset=dataset, engine=api.InMemoryEngine()) as plan,
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
neighborhood = api.NeighborhoodSpec(
    layers=(
        api.GraphLayer(
            samples=(
                api.GraphSample(
                    relation="user_purchases", fanout=5, direction="forward", strategy="recent"
                ),
            )
        ),
        api.GraphLayer(
            samples=(
                api.GraphSample(
                    relation="item_purchases",
                    fanout=1,
                    direction="reverse",
                    strategy="uniform",
                    temporal="static",
                ),
            )
        ),
    )
)
graph_sample = api.SampleSpec(
    entity=api.EntitySpec(table="users", entity_type="user"),
    neighborhoods={"bought": neighborhood},
)
graph_batch = api.BatchSpec(
    inputs={
        "graph": api.Input(
            source=api.NeighborhoodSource(name="bought"),
            representation=api.Graph(
                node_fields={
                    "items": api.Fields(
                        fields=(api.Encode(field="category", encoding=api.Vocabulary()),)
                    ),
                    "purchases": api.Fields(
                        fields=(api.Encode(field="amount", encoding=api.ZScore()),)
                    ),
                }
            ),
            layout=None,
        )
    },
    batching=api.Batching(max_samples=2),
)
graph_contract = api.DataLoaderSpec(schema=schema, sample=graph_sample, batch=graph_batch)

# The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
with (
    api.prepare(contract=graph_contract, dataset=dataset, engine=api.InMemoryEngine()) as plan,
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

- A timed destination is filtered by its clock; a clockless one must say
  `temporal="static"`, so you never read a table without its time filter by
  mistake.
- `strategy="recent"`, the default, keeps the newest rows and needs a clock,
  even for a static traversal. `"uniform"` draws a deterministic random sample,
  but only on the in-memory engine; the DiskGraph engine refuses it at
  `check()`. So on DiskGraph a traversal can't reach a clockless table like
  `items` here; give the table a clock if you need to sample it there (see
  [limitations](limitations.md#time)).
- Each example's graph is disjoint: item 10 appears once for each user that
  reached it. Edges keep the relation's declared left-to-right orientation even
  when traversed in reverse.

## Caller-supplied candidates

For ranking, the caller chooses which items to score; a `CandidateSetSpec`
says what to read for each candidate at the request's prediction time, and
labels align with the caller's order:

```python
candidate_sample = api.SampleSpec(
    entity=api.EntitySpec(table="users", entity_type="user"),
    candidate_sets={
        "items": api.CandidateSetSpec(
            entity=api.EntitySpec(table="items", entity_type="item"),
            values={
                "category": api.Field(column="category"),
                "prior_buyers": api.Aggregate(
                    source=api.Rows(
                        path=("item_purchases",), time=api.TimeWindow(column="at", stop=0)
                    ),
                    operation="count",
                ),
            },
            targets={"bought": api.ExternalTarget(dtype=api.DType(name="bool"))},
        )
    },
)
candidate_batch = api.BatchSpec(
    inputs={
        "candidates": api.Input(
            source=api.CandidateSource(name="items"),
            representation=api.Fields(
                fields=(
                    api.Encode(field="category", encoding=api.Vocabulary()),
                    api.Encode(field="prior_buyers", encoding=api.Raw()),
                )
            ),
            layout=api.Padded(),
        )
    },
    targets={
        "bought": api.CandidateTarget(input="candidates", source="bought", encoding=api.Raw())
    },
    batching=api.Batching(max_samples=1),
)
candidate_request = api.SampleRequest(
    sample_id="user-1",
    entity_id=1,
    as_of_time=10 * DAY,
    candidates={
        "items": tuple(
            api.CandidateRequest(
                entity_id=item, targets=(api.TargetValue(name="bought", value=item == 12),)
            )
            for item in (10, 11, 12)
        )
    },
)
# The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
with (
    api.prepare(
        contract=api.DataLoaderSpec(schema=schema, sample=candidate_sample, batch=candidate_batch),
        dataset=dataset,
        engine=api.InMemoryEngine(),
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
    columns={"id": api.DType(name="int64"), "age": api.DType(name="int64")},
    time=api.NON_TEMPORAL,
)
built = build.contract(
    schema=build.Schema(tables=(users,)),
    sample=build.Sample(entity=users, values={"age": build.Read(column=users.age)}),
    batch=build.Batch(
        inputs={
            "profile": build.Input(
                source=users,
                representation=api.Fields(fields=(api.Encode(field="age", encoding=api.Raw()),)),
                layout=api.Separate(),
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
