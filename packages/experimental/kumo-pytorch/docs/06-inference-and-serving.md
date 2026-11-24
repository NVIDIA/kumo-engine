# 6. Inference and serving

A model only works if the batches it scores look like the batches it was
trained on: the same vocabulary IDs, the same normalization, the same history
cut at the same point in time. If your scoring job or your online service builds
its inputs a little differently, the model still returns numbers, and they are
wrong.

This page walks one small example from training to batch scoring to online
serving, and shows how to keep the inputs identical at every step.

kumo-pytorch gives you three things for this:

- An artifact, a directory that saves the contract and the fitted processing
  state (vocabularies, normalization), so another process builds the same
  inputs. You supply the dataset and anchors, and you save model weights
  separately with `api.save_weights`.
- An inference mode that builds the same inputs without computing labels.
- A serving assembler that uses a prepared DiskGraph index and Parquet anchors
  to produce the same model-format inputs as offline inference.

You own the model, training loop, checkpoints, serving index and deployment.

```text
TRAINING JOB                  SAVED TO DISK              BATCH SCORING JOB
fit on training data ───────► run-001/artifact/ ──┬────► load artifact + weights
train your model              run-001/model.pt    │      score many users at once
                                                  │
                                                  │      ONLINE SERVICE
                                                  └────► load artifact + weights, once
                                                         prepare the DiskGraph index once
                                                         per call: assemble(anchors); score
```

Every code block on this page runs in order. The blocks marked "in a fresh
process" start a new Python process, as a scoring job or a service would.

## The running example

The example has two tables, `users` and their `purchases`. For each user at a
point in time, the model sees three things:

- the user's **segment** (`loyal`, `casual` or `dormant`);
- **purchases so far**: how many purchases the user has made up to that time;
- the **last 8 purchases**, each with its amount and category.

The label is **how many purchases the user makes in the next 7 days**. The
model predicts whether that count is above zero.

Each sample starts from an **anchor**: a `SampleRequest` with a sample ID, the
user's ID, and the prediction time (`as_of_time`), which this page calls the
**cutoff**. Everything the model sees is read from rows at or before the
cutoff; the label is read from rows after it.
The example trains on anchors at day 20 and scores anchors at day 34.

The training job, the scoring job and the service all import the same module.
Save it as `shop.py`: the synthetic data, the contract (a `DataLoaderSpec`, see
[Contracts](02-contracts.md)), a small PyTorch model sized from the fitted
state, and its example requests.

<!-- test: file shop.py -->
```python
import random
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from kumo.pytorch import api

DAY = 86_400
TRAIN_AT, SCORE_AT = 20 * DAY, 34 * DAY
SEGMENTS = ("loyal", "casual", "dormant")
RATES = {"loyal": 0.8, "casual": 0.25, "dormant": 0.03}
CATEGORIES = ("books", "games", "music")
Tables = dict[str, list[dict[str, Any]]]


def tables() -> Tables:
    """Synthetic users whose segment sets how often they buy: the planted signal."""
    rng = random.Random(0)
    users, purchases = [], []
    for user_id in range(1, 61):
        segment = SEGMENTS[user_id % 3]
        users.append({"user_id": user_id, "segment": segment})
        for day in range(40):
            if rng.random() < RATES[segment] / 3:
                purchases.append(
                    {
                        "purchase_id": len(purchases) + 1,
                        "user_id": user_id,
                        "at": day * DAY + rng.randrange(DAY),
                        "amount": round(rng.uniform(2.0, 60.0), 2),
                        "category": rng.choice(CATEGORIES),
                    }
                )
    return {"users": users, "purchases": purchases}


def contract() -> api.DataLoaderSpec:
    users = {"user_id": api.DType("int64"), "segment": api.DType("string")}
    purchases = {
        "purchase_id": api.DType("int64"),
        "user_id": api.DType("int64"),
        "at": api.DType("timestamp", unit="s", timezone="UTC"),
        "amount": api.DType("float64"),
        "category": api.DType("string"),
    }
    schema = api.DataSchema(
        tables=(
            api.TableSpec("users", "user_id", users, time_column=api.NON_TEMPORAL),
            api.TableSpec("purchases", "purchase_id", purchases, time_column="at"),
        ),
        relations=(api.RelationSpec("user_purchases", "users", "user_id", "purchases", "user_id"),),
    )
    history = api.Rows(("user_purchases",), api.TimeWindow("at", stop=0))  # up to the cutoff
    next_week = api.Rows(("user_purchases",), api.TimeWindow("at", start=0, stop=7 * DAY))
    sample = api.SampleSpec(
        entity=api.EntitySpec("users", entity_type="user"),
        entity_values={
            "segment": api.Field(column="segment"),
            "purchases_so_far": api.Aggregate(history, "count", dtype=api.DType("int64")),
        },
        events={
            "purchases": api.EventSpec(
                history,
                fields={
                    "amount": api.Field(column="amount"),
                    "category": api.Field(column="category"),
                },
                limit=8,
                order="descending",
            )
        },
        # Read from rows after the cutoff: computed in training, never in inference.
        targets={
            "purchases_next_week": api.Aggregate(next_week, "count", dtype=api.DType("int64"))
        },
    )
    batch = api.BatchSpec(
        inputs={
            "user": api.Input(
                api.EntitySource(),
                api.Fields(
                    (
                        api.Encode("segment", api.Vocabulary()),
                        api.Encode("purchases_so_far", api.ZScore()),
                    ),
                    row=api.Columns(),
                ),
                api.Separate(),
            ),
            "recent": api.Input(
                api.EventSource(("purchases",)),
                api.Fields(
                    (api.Encode("amount", api.ZScore()), api.Encode("category", api.Vocabulary())),
                    row=api.Columns(),
                ),
                api.Padded(),
            ),
        },
        targets={"active": api.LabelTarget("purchases_next_week", api.Raw())},
        batching=api.Batching(max_samples=16),
    )
    return api.DataLoaderSpec(schema, sample, batch)


class ActivityModel(nn.Module):
    """A user model over the batch as the package emits it, sized from the fitted state."""

    def __init__(self, output_schema: Mapping[str, Any]) -> None:
        super().__init__()
        user = output_schema["inputs"]["user"]
        recent = output_schema["inputs"]["recent"]
        self.segment = nn.Embedding(user["segment"]["vocabulary_size"], 8)
        self.category = nn.Embedding(recent["category"]["vocabulary_size"], 8)
        self.head = nn.Sequential(nn.Linear(8 + 1 + 8 + 1, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, batch: api.Batch) -> torch.Tensor:
        user, recent = batch.inputs["user"], batch.inputs["recent"]
        valid = recent["valid"].unsqueeze(-1).float()
        events = torch.cat(
            (
                self.category(recent["category"]["values"]),
                recent["amount"]["values"].unsqueeze(-1),
            ),
            dim=-1,
        )
        pooled = (events * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        features = torch.cat(
            (
                self.segment(user["segment"]["values"]),
                user["purchases_so_far"]["values"].unsqueeze(-1),
                pooled,
            ),
            dim=-1,
        )
        return self.head(features).squeeze(-1)


def anchors(at: int, *, split: str | None) -> list[api.SampleRequest]:
    return [
        api.SampleRequest(f"user-{user_id}-at-{at // DAY}", user_id, at, split=split)
        for user_id in range(1, 61)
    ]


def dataset(rows: Tables) -> api.Dataset:
    return api.Dataset({name: api.InMemoryRows(values) for name, values in rows.items()})
```

`user` and `recent` are the input names you will see in every batch. `active`
is the label.

## Train and save

```python
from pathlib import Path

import torch
from torch import nn

from kumo.pytorch import api
from shop import TRAIN_AT, ActivityModel, anchors, contract, dataset, tables

workdir = Path("run-001")
torch.manual_seed(0)
# The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
with api.prepare(contract(), dataset(tables()), api.InMemoryEngine()) as plan:
    with plan.materialize(anchors(TRAIN_AT, split="train"), fit_split="train") as task:
        model = ActivityModel(task.output_schema)  # embedding sizes come from here
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        for epoch in range(30):
            for batch in task.batches(split="train", epoch=epoch, shuffle=True):
                labels = (batch.targets["active"]["values"] > 0).float()
                loss = nn.functional.binary_cross_entropy_with_logits(model(batch), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        task.to_artifact().save(workdir / "artifact")
        api.save_weights(workdir / "model.pt", model.state_dict(), task.fitted_state)
```

`fit_split="train"` tells the package which anchors it may learn from. What it
learns is the **fitted state**: everything computed from training data that
shapes the inputs. Here that is the segment and category vocabularies (which
string gets which integer ID) and the mean and scale used to normalize the
purchase count and the amount.

The **artifact** is the contract plus its fitted state and its identity, saved
together in one directory:

| File | Content |
| --- | --- |
| `contract.json` | The contract the task was materialized from. |
| `fitted-state.json` | The fitted values, readable, and where they were fitted (the `fit` record). |
| `task.json` | The task's fingerprints and output schema. |
| `manifest.json` | Summaries, fingerprints, and the SHA-256 digest of every file. |

`task.to_artifact()` builds it in memory; `Artifact.save` writes it. The files
belong together: don't edit them, and don't mix files from different saves.

Save the weights with `save_weights`, not a bare `torch.save`. Embedding rows are
indexed by the vocabulary IDs in the fitted state: row 4 of the segment
embedding means `loyal` only under this fit. `save_weights` records which
fitted state the weights belong to, so they can't be loaded against another.

## Load in a fresh process

Load the artifact in your scoring job or service. Nothing is refitted and no
data is read: you get back exactly the preprocessing the model was trained with.

<!-- test: fresh-process -->
```python
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

from kumo.pytorch import api
from shop import SCORE_AT, ActivityModel, anchors, dataset, tables

workdir = Path("run-001")
artifact = api.Artifact.load(workdir / "artifact")
model = ActivityModel(artifact.output_schema)
model.load_state_dict(api.load_weights(workdir / "model.pt", artifact.fitted_state))
model.eval()
print(artifact.output_schema["inputs"]["user"]["segment"]["vocabulary_size"])
```

<!-- test: output -->
```text
5
```

What you get back:

- `artifact.contract`: the training contract. Use it for everything that
  follows.
- `artifact.fitted_state`: the fitted state, ready to pass as `fitted=`.
- `artifact.output_schema`: sizes and encodings of every output, to build the
  model with. The segment vocabulary has 5 IDs: three segments plus the IDs
  reserved for a missing and an unknown value.

`Artifact.load` refuses the directory, and names the file, when a file is
missing, was changed after it was saved or comes from a different save, or when
the files don't agree with each other:

```text
ValueError: run-001/artifact/fitted-state.json does not match the digest manifest.json records for it; the artifact's files were not written together
```

The digests catch files that were not written together or were edited
separately; they are not a signature against deliberate tampering. The full
list of checks is under [`Artifact.load`](api/artifacts-and-serving.md#artifactload).
`load_weights` likewise refuses weights saved for a different contract or a
different fitted state.

Contracts, fitted state and artifacts can be pickled, so you can pass them to
DataLoader workers or other processes, `spawn` included, instead of reloading
the artifact in each.

## Batch scoring

Score a set of anchors in one job. Prepare the artifact's contract over your
scoring data, and materialize with the artifact's fitted state and
`inference=True`:

```python
def leaves(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, inner in value.items():
            yield from leaves(f"{prefix}.{key}", inner)
    else:
        yield prefix, value


predictions = {}
# The in-memory engine is for trying things out on tiny data; use DiskGraph for real data.
with api.prepare(artifact.contract, dataset(tables()), api.InMemoryEngine()) as plan:
    scoring = anchors(SCORE_AT, split=None)
    with plan.materialize(scoring, fitted=artifact.fitted_state, inference=True) as task:
        with torch.no_grad():
            for batch in task.batches():
                if not predictions:
                    for path, tensor in leaves("inputs", batch.inputs):
                        print(f"{path:<34} {tuple(tensor.shape)} {tensor.dtype}")
                    print("targets", batch.targets)
                    print("sample_ids[:3]", batch.metadata["sample_ids"][:3])
                scores = torch.sigmoid(model(batch)).tolist()
                predictions.update(zip(batch.metadata["sample_ids"], scores, strict=True))
(workdir / "predictions.json").write_text(json.dumps(predictions))
```

The first batch:

<!-- test: output -->
```text
inputs.recent.amount.values        (16, 8) torch.float32
inputs.recent.amount.valid         (16, 8) torch.bool
inputs.recent.category.values      (16, 8) torch.int64
inputs.recent.category.valid       (16, 8) torch.bool
inputs.recent.valid                (16, 8) torch.bool
inputs.user.segment.values         (16,) torch.int64
inputs.user.segment.valid          (16,) torch.bool
inputs.user.purchases_so_far.values (16,) torch.float32
inputs.user.purchases_so_far.valid (16,) torch.bool
targets {}
sample_ids[:3] ('user-1-at-34', 'user-2-at-34', 'user-3-at-34')
```

The inputs have the same names, shapes and encodings as in training. `targets`
is empty. Each row of the batch is identified by `metadata["sample_ids"]`, so
key your predictions by sample ID, as the loop does. The metadata also carries
`entity_ids`, `as_of_time`, `splits`, `is_padding` and the contract and artifact
fingerprints.

### Batch scoring on the DiskGraph engine

The example above scores on the in-memory engine to keep it small. A real
scoring job reads Parquet files through the DiskGraph engine, and its anchors
are a Parquet `AnchorTable`. The artifact, the weights and the model stay the same.
Here we write the example's tables and scoring anchors to Parquet and score
them:

<!-- test: diskgraph -->
```python
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq

data = Path("shop-data")
data.mkdir(exist_ok=True)
rows = tables()
pq.write_table(pa.Table.from_pylist(rows["users"]), data / "users.parquet")
purchase_schema = pa.schema(
    [
        ("purchase_id", pa.int64()),
        ("user_id", pa.int64()),
        ("at", pa.timestamp("s", tz="UTC")),
        ("amount", pa.float64()),
        ("category", pa.string()),
    ]
)
purchases = pa.Table.from_pylist(rows["purchases"], schema=purchase_schema)
pq.write_table(purchases, data / "purchases.parquet")
score_time = datetime.fromtimestamp(SCORE_AT, UTC)
scoring_rows = [
    {"sample_id": a.sample_id, "user_id": a.entity_id, "as_of": score_time}
    for a in anchors(SCORE_AT, split=None)
]
pq.write_table(pa.Table.from_pylist(scoring_rows), data / "scoring-anchors.parquet")

parquet = api.Dataset(
    {name: api.Parquet(path=str(data / f"{name}.parquet")) for name in ("users", "purchases")}
)
scoring_table = api.AnchorTable(
    source=api.Parquet(path=str(data / "scoring-anchors.parquet")),
    entity_id="user_id",
    as_of_time="as_of",
    sample_id="sample_id",
)
engine = api.DiskGraphEngine(index=data / "index")
disk_predictions = {}
with api.prepare(artifact.contract, parquet, engine) as plan:
    with plan.materialize(scoring_table, fitted=artifact.fitted_state, inference=True) as task:
        with torch.no_grad():
            for batch in task.batches(output_format="model"):
                scores = torch.sigmoid(model(batch)).tolist()
                disk_predictions.update(zip(batch.metadata["sample_ids"], scores, strict=True))
print(len(disk_predictions), "anchors scored")
print(f"user-3-at-34 scores {disk_predictions['user-3-at-34']:.3f}")
```

<!-- test: output -->
```text
60 anchors scored
user-3-at-34 scores 0.647
```

The first `prepare` builds the index under `shop-data/index`. A scheduled
scoring job usually passes `ingest="never"`, so that it fails if the index it
expects is missing instead of building a new one. See
[building and reusing the index](04-engines-and-data.md#building-and-reusing-the-index).

### What inference mode changes

Inference mode builds the model's inputs and nothing else. Labels are not
computed, so the rows only a label would read are never read. In the example,
the data runs to day 40. A training task anchored at day 34 would read the
purchases of the following 7 days to compute its label; the inference task
never reads them. You can score anchors whose future hasn't happened yet, and
rows dated after the cutoff can't change a score, except that on the DiskGraph
engine they can change which tied neighbor a graph sample keeps (see [scoring
offline on the DiskGraph engine](#scoring-offline-on-the-diskgraph-engine)). Anchors need no labels;
labels an anchor table supplies are ignored.

| Target in the contract | In inference mode |
| --- | --- |
| `LabelTarget`, `StructuredTarget`, `CandidateTarget` | Not computed and not emitted. The rows they read are never read. |
| `ExternalTarget` (labels you supply) | Not emitted. |
| `MaskedValuesTarget` | Masking is off. The model sees the real values. |
| `NextTokenTarget` | Not emitted. The sequence runs to the cutoff with no final token held back, so the model predicts the token after the last one. |
| `MatchingGroupTarget`, `AlignedEventTarget` | Not emitted. Fields an aligned target hides from the inputs stay hidden. |

Every anchor is scored. The last batch can be short (the example's 60 anchors
give batches of 16, 16, 16 and 12), even if the contract trains with
`drop_last`. Pass `drop_last=True` to `task.batches` only if you really want to
skip the remainder.

Inference never fits: it requires `fitted=` and refuses `fit_split`. `fitted`
is either the complete state, for the same contract, or its input half,
`fitted_state.inputs`, for a contract with the same inputs and any targets (see
[Passing only the input half](#passing-only-the-input-half-of-the-fitted-state)).

### Scoring when the DiskGraph engine can't build the label

`prepare` checks the whole contract against the engine, labels included,
because the plan may train. A label can need more than its inputs: for example a
filter on the label rows, or a sum over a decimal column. The DiskGraph engine
refuses such a label even though inference would never compute it. To score only, say so
when you prepare:

<!-- test: diskgraph -->
```python
with api.prepare(artifact.contract, parquet, engine, inference=True) as plan:
    with plan.materialize(scoring_table, fitted=artifact.fitted_state) as task:
        print(task.count(), "anchors to score")
```

<!-- test: output -->
```text
60 anchors to score
```

`check(..., inference=True)` and `prepare(..., inference=True)` ask the engine
only about what inference runs: the inputs. The contract must still be well
formed. The plan keeps the training contract and its fingerprints, so the
artifact's fitted state and the weights still apply. Its tasks are inference
tasks: `materialize` defaults to `inference=True` and refuses `inference=False`.
To train, prepare without it.

## Online serving

Prepare the artifact's contract against the DiskGraph dataset when the service
starts. Keep that plan open while handling requests. Each call takes a Parquet
anchor table identifying the samples, entities and cutoffs to score. The
assembler uses the artifact's fitted state and returns model-format batches
from that index. The caller owns the plan, the index and the anchor files.

Here, the caller selects a single request from the scoring anchors written
above and compares its prediction with batch scoring on the same index:

<!-- test: diskgraph -->
```python
request_path = data / "request.parquet"
pq.write_table(pq.read_table(data / "scoring-anchors.parquet").slice(2, 1), request_path)
request = api.AnchorTable(api.Parquet(str(request_path)), "user_id", "as_of", sample_id="sample_id")
with api.prepare(artifact.contract, parquet, engine, inference=True) as plan:
    server = api.ServingAssembler(artifact, plan=plan)
    (batch,) = server.assemble(request)
    with torch.no_grad():
        score = torch.sigmoid(model(batch)).item()
(sample_id,) = batch.metadata["sample_ids"]
assert abs(score - disk_predictions[sample_id]) < 1e-6
```

`ServingAssembler(artifact).rows_spec` is available for inspecting the tables,
columns, windows and limits the contract reads. A row dictionary and a list of
`SampleRequest` objects are refused by `assemble`. To change the source data,
prepare a plan over the updated dataset and index.

## Pitfalls

### Refitting at scoring time

If you materialize the scoring anchors without `fitted=`, scores come out
wrong, or `load_weights` refuses the task's state. Without `fitted=`,
`materialize` fits again, on the scoring anchors.
Vocabulary IDs follow how often each value occurs in the fitting data, so they
can change, and normalization statistics do. The weights were trained against
the old values. If you load the weights against `artifact.fitted_state` and then
score batches from a refitted task, nothing stops you, and the scores are
wrong.

Always pass `fitted=artifact.fitted_state` together with `inference=True`.
Inference mode never fits, so it refuses to run without fitted state:

```text
ValueError: inference never fits: pass fitted= with the training task's state, for example Artifact.load(directory).fitted_state
```

### Rebuilding the contract in code

If `materialize` refuses the fitted state with "fitted state was fitted for
different inputs", you've probably rebuilt the contract from code. A
contract's identity is a fingerprint of its content. Calling your
`contract()` function again works only while the code is unchanged. Once someone
edits it (say `limit=8` becomes `limit=16`), the fingerprint differs and the
fitted state no longer applies.

Score and serve with `artifact.contract` instead. It is the contract the
model was trained with, whatever the code says today.

### Keying predictions by position

Predictions end up attached to the wrong users if you match them by position.
Batch order follows the anchors and the batch arguments, not your lookup
table. In a multi-rank run, ranks also get the same number of anchors: a
short rank repeats one of its anchors and marks the copy in
`metadata["is_padding"]` (see [Training batches](05-training-batches.md)).

Key every prediction by `metadata["sample_ids"]` and skip rows where
`is_padding` is true:

<!-- test: skip (fragment: runs inside an inference task on one rank) -->
```python
for batch in task.batches(rank=rank, world_size=world_size):
    scores = torch.sigmoid(model(batch)).tolist()
    padding = batch.metadata["is_padding"].tolist()
    for sample_id, score, pad in zip(batch.metadata["sample_ids"], scores, padding, strict=True):
        if not pad:
            predictions[sample_id] = score
```

Don't assume every batch is full either: the last one is kept even when it is
short.

### Passing only the input half of the fitted state

`fitted_state.inputs` is the fitted state without its target half. It reuses
input vocabularies under a contract with other targets: to fine-tune on a new
label, pass it to `materialize` and let the targets fit there (see
[Training batches](05-training-batches.md)). Inference accepts it too and gives
the same inputs. For example, a pre-trained encoder's input half can embed
entities under a contract with the same inputs and no targets.

Two things differ from passing the complete state:

- the task's `fitted_state` is the input half, so `task.to_artifact()` refuses:
  an artifact holds complete state. Keep the training task's artifact;
- weights are tied to the state they were saved with, so `load_weights`
  refuses weights saved under `artifact.fitted_state` when given
  `fitted_state.inputs`. Load them with the complete state.

### Scoring offline on the DiskGraph engine

Use the same prepared index, anchors, batch settings, seed and artifact when
comparing served batches with offline DiskGraph inference. Rebuilding the index
can change which tied graph neighbor a fanout selects. See
[engine support](limitations.md#engine-support).

## Next

- [Model input shapes](07-model-input-shapes.md): four published sequence-model
  batch shapes over one schema.
- [Supported features and limitations](limitations.md): what serving and
  artifacts do not do yet.
- [API reference: artifacts and serving](api/artifacts-and-serving.md).
