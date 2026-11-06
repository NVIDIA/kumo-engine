# kumo-pytorch

> This package is developed in a separate repository and exported here, so
> pull requests that change this directory can't be merged directly. To report
> a problem or suggest a change, open an issue in this repository.

kumo-pytorch is an experimental PyTorch library for building model inputs from
relational tables, for both training and inference. Expect the API to change
between releases, and see [limitations](docs/limitations.md) for what isn't
supported yet. Please report problems and questions as
[kumo-engine issues](https://github.com/NVIDIA/kumo-engine/issues).

kumo-pytorch prepares the data for PyTorch models that learn from relational
data: the users, accounts, transactions and products tables a business keeps in
its databases. You describe what one example looks like, and kumo-pytorch does
the work between your tables and the batches your model consumes, whether it's
training, scoring a large batch offline, or answering a single request online.

That work usually takes a lot of custom code. For every example, you have to
gather the right rows from several tables as they were at the example's
prediction time, compute features from them, turn categories, numbers and text
into IDs and tensors using what was learned from the training data, and pack
histories of different lengths into batches. kumo-pytorch handles all of it from
one description, along this path:

```text
tables ─▶ index ────────▶ featurize ──────▶ fit ───────────▶ sample ─────▶ encode & pack ─▶ deliver
          your data       point-in-time     train split:     histories,    tokenize,        batches
          in DiskGraph    aggregates,       vocabularies,    graph         pad or pack      per GPU
                          derived values    buckets,         neighbors     under a          rank
                                            tokenizers                     budget
```

In a bit more detail:

- Each example only sees rows from at or before its
  prediction time, including rows it reaches through other tables, so your model
  can't learn from the future by accident.
- Counts and sums over time windows, the time since a user's last
  purchase, calendar parts and similar values are declared next to the raw
  columns.
- An example can include a sequence of past events,
  such as a user's transactions, or a neighborhood of related rows sampled a
  few hops out.
- Vocabularies, numeric buckets, normalization
  and text tokenizers are fitted on the training split and saved, so evaluation
  and inference encode values the same way training did.
- Inputs come out as named tensors, padded or packed under a
  sample or token budget, in a repeatable order, and split across GPUs when you
  train on more than one.
- The same description builds batches for
  training, evaluation, batch scoring and single requests at serving time.

kumo-pytorch doesn't require a particular model. You bring the model, loss and
training loop, and it hands you the batches. The `models` and `architectures`
modules hold reference encoders that the package's own tests use; they aren't a
stable API.

Batches are built by kumo-engine's Rust-based DiskGraph engine, which indexes
data on local SSDs and retrieves time-correct neighborhoods for training and
batch prediction, without keeping the full graph index in RAM. The
[kumo-engine README](https://github.com/NVIDIA/kumo-engine#readme) has more on
how it works.

There's also a small in-memory engine that works on Python lists of rows. The
guide uses it to keep examples tiny and easy to print; it isn't meant for real
data.

## How it works

Say you want to predict whether a user will buy something next week. Each
training example is one user at one point in time. To build these examples,
kumo-pytorch needs three things from you:

- A **schema** (`DataSchema`) that describes your tables, how they connect, and
  which column holds each row's timestamp.
- A **sample spec** (`SampleSpec`) that says what an example looks at, for
  instance the user's country and their past purchases.
- A **batch spec** (`BatchSpec`) that says how those values are encoded and laid
  out as tensors for your model.

Together these make up the contract. You then pass in the examples you want,
each a user ID and a prediction time, and get back batches. The fitted encoders
are saved with the contract, so you can build the same inputs again later in
another process.

## Install

kumo-pytorch isn't on PyPI. You build it, together with the DiskGraph engine it
depends on, from a checkout of
[NVIDIA/kumo-engine](https://github.com/NVIDIA/kumo-engine). You'll need Python
3.11 or newer, [uv](https://docs.astral.sh/uv/), a Rust toolchain and C/C++
build tools:

```bash
git clone https://github.com/NVIDIA/kumo-engine.git
cd kumo-engine
uv build --wheel --out-dir dist packages/diskgraph
uv build --wheel --package kumo-pytorch --out-dir dist
uv venv --python 3.12 .venv-kumo-pytorch
uv pip install --python .venv-kumo-pytorch/bin/python --torch-backend cpu \
    dist/diskgraph-*.whl dist/kumo_pytorch-*.whl
```

Use `--torch-backend cu130` for the CUDA 13.0 build of PyTorch. The
[installation](docs/installation.md) page covers pip, the notebook
dependencies, and running the tests.

## Quick start

Here are two small tables, users and purchases, saved as Parquet files. We
build one batch for two users, each as of January 10.

<!-- test: diskgraph -->
```python
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kumo.pytorch import api

# Some data: two users and their purchases, saved as Parquet files.
data = Path(tempfile.mkdtemp())
users = [{"user_id": 1, "country": "US"}, {"user_id": 2, "country": "GB"}]
purchases = [
    {"purchase_id": 101, "user_id": 1, "at": datetime(2024, 1, 2), "amount": 12.0},
    {"purchase_id": 102, "user_id": 1, "at": datetime(2024, 1, 8), "amount": 30.0},
    {"purchase_id": 103, "user_id": 1, "at": datetime(2024, 1, 12), "amount": 99.0},
    {"purchase_id": 201, "user_id": 2, "at": datetime(2024, 1, 6), "amount": 18.0},
]
pq.write_table(pa.Table.from_pylist(users), data / "users.parquet")
purchases_schema = pa.schema(
    [
        ("purchase_id", pa.int64()),
        ("user_id", pa.int64()),
        ("at", pa.timestamp("s", tz="UTC")),
        ("amount", pa.float64()),
    ]
)
pq.write_table(pa.Table.from_pylist(purchases, schema=purchases_schema), data / "purchases.parquet")

# The two tables and how they connect. Column types are read from the files.
schema = api.DataSchema(
    tables=(
        api.TableSpec(name="users", primary_key="user_id", time_column=api.NON_TEMPORAL),
        api.TableSpec(name="purchases", primary_key="purchase_id", time_column="at"),
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

# What each example looks at: the user's country and their earlier purchases.
past = api.Rows(path=("user_purchases",), time=api.TimeWindow(column="at", stop=0))
sample = api.SampleSpec(
    entity=api.EntitySpec(table="users", entity_type="user"),
    entity_values={"country": api.Field(column="country")},
    events={"purchases": api.EventSpec(rows=past, fields={"amount": api.Field(column="amount")})},
)

# How the model sees it: the country as a vocabulary index, and the purchase
# amounts normalized and padded into a sequence.
batch = api.BatchSpec(
    inputs={
        "user": api.Input(
            source=api.EntitySource(),
            representation=api.Fields(
                fields=(api.Encode(field="country", encoding=api.Vocabulary()),)
            ),
            layout=api.Separate(),
        ),
        "history": api.Input(
            source=api.EventSource(events=("purchases",)),
            representation=api.Fields(fields=(api.Encode(field="amount", encoding=api.ZScore()),)),
            layout=api.Padded(),
        ),
    },
    batching=api.Batching(max_samples=2),
)

# Two examples to build: users 1 and 2, both as of January 10.
now = datetime(2024, 1, 10, tzinfo=timezone.utc)
requests = (
    api.SampleRequest(sample_id="user-1", entity_id=1, as_of_time=now),
    api.SampleRequest(sample_id="user-2", entity_id=2, as_of_time=now),
)

# Put it together. The DiskGraph engine indexes the files the first time, then builds the batch.
contract = api.DataLoaderSpec(schema=schema, sample=sample, batch=batch)
dataset = api.Dataset(
    tables={
        "users": api.Parquet(path=str(data / "users.parquet")),
        "purchases": api.Parquet(path=str(data / "purchases.parquet")),
    }
)
engine = api.DiskGraphEngine(index=data / "index", ingest="if_missing")
with (
    api.prepare(contract=contract, dataset=dataset, engine=engine) as plan,
    plan.materialize(anchors=requests) as task,
):
    result = next(task.batches(requests=requests))

print("inputs:", tuple(result.inputs))
print("history amount shape:", tuple(result.inputs["history"]["amount"]["values"].shape))
print("history valid rows:", result.inputs["history"]["valid"].tolist())
```

<!-- test: output -->
```text
inputs: ('user', 'history')
history amount shape: (2, 2)
history valid rows: [[True, True], [True, False]]
```
User 1 also bought something on January 12, but that's after the prediction
time, so only their two earlier purchases show up. User 2 has just one purchase,
so the second slot in their history is padding and is marked as not valid.

## Learn more

- The [guide](docs/README.md) walks through the concepts, engines and data,
  training and serving, in reading order.
- The [examples](examples/README.md) are three end-to-end notebooks on generated
  data: product recommendation, fraud detection and churn prediction.
- The [API reference](docs/api/README.md) documents every public class and
  function.
- The [behavior guarantees](conformance/README.md) are the rules the DiskGraph
  engine is tested against, each backed by tested examples. Where it doesn't
  meet one yet, the [limitations](docs/limitations.md) page says so.
