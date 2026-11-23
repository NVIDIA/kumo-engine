# Relational pretraining with Kumo Engine

Feed Blueprint v2 directly from relational tables using Kumo Engine's DiskGraph
loader. The API defines which rows form a sample, how features are computed and
encoded, and how those values become model batches. DiskGraph retrieves histories
on demand while the model trains.

## Relational input

[seed_data.py](seed_data.py) converts the Blueprint's
[Pragmatiq Gen2 book](../../configs/data/pragmatiq/world.yaml) into five Parquet
tables and repeated observation anchors:

```text
users (user_id, country, age_band, occupation, ...)
  ├── transactions  (event_id, user_id, ts, amount, merchant, ...)
  ├── app_events    (event_id, user_id, ts, screen, action, ...)
  ├── trades        (event_id, user_id, ts, instrument, quantity, ...)
  └── communications (event_id, user_id, ts, campaign_id, channel, ...)

anchors (sample_id, user_id, as_of_time, split)
```

`users.user_id` is the entity key. Each event has its own primary key, a foreign
key to the user and a timestamp. Multiple anchors for one user produce samples
at different observation times. The default book contains 100,000 users across
31 months; its records are synthetic.

## Define the history

[contract.py](contract.py) declares the schema and traverses each event relation
relative to an anchor. For example, transaction history is selected with:

```python
api.Rows(
    ("user_transactions",),
    api.TimeWindow("ts", start=-730 * 86400, stop=0),
)
```

One `EventSource` combines all four streams, keeps the latest 128 events
globally and returns them oldest first:

```python
api.EventSource(
    events=("transactions", "app_events", "trades", "communications"),
    max_events=128,
    keep="last",
    order="ascending",
)
```

Retrieval, temporal filtering and merging run natively in DiskGraph. The
contract describes the selection once; callers supply the anchors to sample.

## Compute and encode features

Features are declared beside the sampled fields. Calendar parts and event ages
come from `Derived(CalendarPart(...))` and `EventTimeDelta`. An aggregate computes
the user's first transaction over all prior transaction history:

```python
first_transaction = api.GroupBy(
    source=api.Rows(("user_transactions",), api.TimeWindow("ts", stop=0)),
    keys=("user_id",),
    values={"first_at": api.Reduce(operation="min", column="ts")},
)
seconds_since_first = api.Aggregate(
    source=first_transaction, operation="recency", column="first_at"
)
```

The API returns anchor time minus `first_at` in seconds. This feature uses all
prior transactions, independently of the decoder's bounded history.

The contract also declares categorical vocabularies, zero-separated numeric
buckets and a shared model ID space. `IdSpace.special_ids` assigns Blueprint's
IDs directly: pad `0`, mask `1`, unknown `2`, entity `3`, event `4`, missing `5`.
Timestamped encoders fit at the training cutoff; the fitted state is saved for
consistent encoding when reloaded.

## Shape and deliver batches

`BatchSpec` declares nested padded event tokens, padded profile tokens and raw
feature columns. `EventsOf(input="history")` keeps calendar/time features aligned
with the merged history.

| Input | Model representation |
|---|---|
| History tokens | `[B,N,T]` key IDs, value IDs, positions and masks |
| Calendar/time features | `[B,N]` hour, weekday, day and relative times |
| Profile tokens | `[B,P]` keys, values and masks |
| First-transaction age | Log-time feature on the corresponding profile token |

[adapter.py](adapter.py) gathers Kumo's event-token rows through its batch event
indices and applies Blueprint's log-time transforms in Torch. Token IDs pass
through directly.

[run.py](run.py) consumes the native batch iterator:

```python
task.batches(
    split="train",
    batch_size=128,
    seed=42,
    epoch=0,
    shuffle=True,
    output_format="model",
)
```

Kumo owns shuffling and epoch order. One producer prefetches the next adapted
CPU batch while the existing eager trainer runs the Blueprint-v2 model and its
next-event and MLM objectives. The indexed graph is reusable; an encoded corpus
of training samples is not retained in Python memory.

## Run

Use Python 3.12 with PyTorch 2.9+, Kumo's PyTorch/DiskGraph packages and Pragmatiq
1.1.0 in the project environment. From the repository root:

```bash
uv pip install --upgrade-package diskgraph --upgrade-package kumo-pytorch \
  -e ".[pragmatiq]" -r examples/kumo-engine/requirements.txt
python examples/kumo-engine/seed_data.py
python examples/kumo-engine/run.py
```

The install command follows Kumo's `main` branch and refreshes both Kumo packages
when rerun. To reproduce a particular environment, replace `@main` in both
requirements with the same commit SHA.

Use `seed_data.py --source PATH` for an existing book. The runner reads
[Blueprint-v2's configuration](../../recipes/blueprint_v2/config.yaml); pass
`--config PATH` to adjust model/training settings, including batch size.
`--index PATH` reuses an index and `--anchors PATH` selects an anchor table.
Data, fitted encoders, weights and training logs go under `outputs/kumo-engine/`.

This example uses eight static profile attributes and derived first-transaction
age, UTC-second timestamps, Kumo encodings and an overlapping final history
window. It runs primary pretraining; downstream probes and fine-tuning are
separate. Static attributes represent the supplied profile snapshot.

Pragmatiq is an independent implementation inspired by the PRAGMA paper
(arXiv 2604.08649), not affiliated with or endorsed by Revolut.
