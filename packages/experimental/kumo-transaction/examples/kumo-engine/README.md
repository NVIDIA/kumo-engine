# Native Kumo loaders

For a dataset supplied as keyed tables and relationships, use the
[shared relational recipe](../../recipes/relational_inputs/README.md). It shows
one declaration for sequence, GNN and hybrid, including generated anchors,
independent availability clocks and timed intermediate tables. Its bounded
native graph path resolves canonical ties by retrieving complete candidate sets;
see that recipe's explicit projection and performance limits.

## Canonical recipes

Python remains the default loader. Sequence, graph and hybrid recipes can opt
into `data.loader: kumo_pytorch` in an experimental configuration. From the
repository root, enter the Transaction project. Native loading requires Python
3.12, rustup, a C/C++ compiler and Linux build tools:

```bash
cd packages/experimental/kumo-transaction
uv sync --frozen --extra dev --extra native --extra kumo
mkdir -p outputs/native-loader
uv run python - <<'PYTHON'
from pathlib import Path
import yaml
for recipe in ("blueprint_v1", "blueprint_v2", "pragma_like", "gnn_encoder", "gnn_sequence"):
    config = yaml.safe_load(Path(f"recipes/{recipe}/config.smoke.yaml").read_text())
    config["data"]["loader"] = "kumo_pytorch"
    Path(f"outputs/native-loader/{recipe}.yaml").write_text(yaml.safe_dump(config))
PYTHON
uv run tfm smoke --recipe blueprint_v1 --config outputs/native-loader/blueprint_v1.yaml
uv run tfm smoke --recipe blueprint_v2 --config outputs/native-loader/blueprint_v2.yaml
uv run tfm smoke --recipe pragma_like --config outputs/native-loader/pragma_like.yaml
uv run tfm smoke --recipe gnn_encoder --config outputs/native-loader/gnn_encoder.yaml
uv run tfm smoke --recipe gnn_sequence --config outputs/native-loader/gnn_sequence.yaml
```

For native FutureSubgraph pretraining, copy the recipe's `config.pretrain_smoke.yaml`
and set the same loader field. Native retrieval has no demonstrated end-to-end
advantage here; see the [default-switch evidence requirements](../../docs/migration/native-loader-default.md#evidence-required-before-changing-the-default).

For train, evaluate and inference commands and data configuration, follow the
[graph encoder](../../recipes/gnn_encoder/README.md#commands),
[hybrid](../../recipes/gnn_sequence/README.md#data-assumptions-and-commands), or
[sequence](../../recipes/blueprint_v2/README.md) recipe. No custom loader wiring
is required. Keep or select `data.loader: python` for the default
reference path. Missing native dependencies fail explicitly.

The project pins Kumo Engine main at
`5c3a22cacc43e8de99531693e3b6612f0b34d637`; `kumo-pytorch` still lives in
`packages/experimental/kumo-pytorch`. The published umbrella wheel does not
provide this API. Install through the Transaction project uv lockfile, rather than mixing
published and source native packages. See the
[migration notes](../../docs/migration/native-loader-default.md) for installation
and checkpoint compatibility. A cold native build can take tens of minutes.

### Sequence retrieval

[NativeHistories](../../src/tfm/data/native_history.py) writes a bounded validated
corpus of event keys, owner keys and clocks into a temporary DiskGraph index.
An `AnchorTable` binds each sample ID, owner and exact cutoff. Native batches
retrieve row IDs; the existing TFM selection and tokenizer paths retain window
membership, tied-event ordering, labels, masks and train-only fitting.
`NativeSequenceBatches` retrieves each indexed training minibatch through the
native task iterator. Explicit caches and packing retain their batch builders.
This path supports every sequence tokenizer/composition, rather than replacing
their vocabularies with native encodings.

### Graphs and hybrid histories

[NativeGraphSampler](../../src/tfm/data/native_graph.py) writes the declared
user/event/transfer FK schema, samples historical neighborhoods using DiskGraph,
and calls [from_kumo_graph](../../src/tfm/data/adapters/kumo_graph.py) with explicit
sample IDs, canonical integer root IDs, numeric features and availability clocks.
Native graph tensors are copied before the task closes. Disjoint graphs are
validated before any intentional same-cutoff union for shared training. Hop
counts are built on CPU. The hybrid aligns its natively retrieved history to
the graph anchors using the existing validated hybrid batch contract.

The worked graph path requires static users, one-parent user-to-event FK roles,
whole-second events/cutoffs, event availability equal to event time and no
explicit observation-start bound. Unsupported settings raise an error naming
the Python control. Empty optional tables remain declared empty tensors in the model batch; the
native index requires at least one populated referenced event table. Reverse
traversal to a single immutable FK parent uses `uniform`, with no temporal rows
or alternative parents to select. Temporal event sampling uses native `recent` fanout; tied fanout membership
follows the native index and need not match Python primary-key ordering.
FutureSubgraph targets remain a separate Python reference traversal over the
future window. They never enter the historical input graph. Native aggregate
parity and full-book throughput are not claimed. The canonical runners still
import objects and materialize bounded batches, targets and probes in memory.

## Advanced Blueprint example with native feature encoders

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
1.1.0 in the project environment. From the repository root, enter the independent
Transaction project:

```bash
cd packages/experimental/kumo-transaction
uv sync --frozen --extra dev --extra native --extra pragmatiq
uv run python examples/kumo-engine/seed_data.py
uv run python examples/kumo-engine/run.py
```

This advanced example uses the same pinned source packages as the canonical
recipes. It demonstrates native feature encoders, a distinct encoding contract.

Use `seed_data.py --source PATH` for an existing book. The runner reads
[Blueprint-v2's configuration](../../recipes/blueprint_v2/config.yaml); pass
`--config PATH` to adjust model/training settings, including batch size.
`--scratch-dir PATH` selects the managed index/task/cache root; `--anchors PATH`
selects an anchor table. Each fresh engine allocates its own build identity.
Data, fitted encoders, weights and training logs go under `outputs/kumo-engine/`.

This advanced contract requires each declared stream table to contain rows;
use the canonical recipes for books with empty optional channels.

This example uses eight static profile attributes and derived first-transaction
age, UTC-second timestamps, Kumo encodings and an overlapping final history
window. It runs primary pretraining; downstream probes and fine-tuning are
separate. Static attributes represent the supplied profile snapshot.

Pragmatiq is an independent implementation inspired by the PRAGMA paper
(arXiv 2604.08649), not affiliated with or endorsed by Revolut.

The runner prints `DiskGraph index: PATH` after preparation. Reuse that completed
index with the same scratch root and a retained anchor table:

```bash
# Run from packages/experimental/kumo-transaction.
uv run python examples/kumo-engine/run.py --scratch-dir PATH/TO/SCRATCH \
  --index PATH/TO/SCRATCH/indexes/BUILD_ID --anchors PATH/TO/anchors.parquet \
  --output outputs/kumo-engine/reused
```

Reuse loads the index’s stored build identity and schema; it does not ingest the
original source Parquet tables again. Keep the managed index in its scratch root.
An index outside the selected scratch root is rejected before preparation; this
flag does not import arbitrary external index directories. Restore/select the
original scratch root rather than moving only the index directory.
