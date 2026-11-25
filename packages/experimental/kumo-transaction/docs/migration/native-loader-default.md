# Source-table native recipe loading

Sequence, graph and hybrid recipes default to Kumo Engine's experimental
PyTorch/DiskGraph loader, `data.loader: kumo_pytorch`. The native recipe path is:

```text
source tables and explicit schema
  → source validation and DiskGraph index
  → generated or supplied AnchorTable
  → continuous native iterator with train-fitted feature encoders
  → owned, prefetched CPU tensor adapter
  → model / trainer
```

An anchor identifies a prediction sample, its root and its cutoff. Repeated
roots at different cutoffs remain different samples. Labels and their
availability clocks travel separately from input features.

The Python reference implementation is an explicit alternative:

```yaml
data:
  loader: python
```

Copy a recipe configuration, set that field, and pass the copy with
`tfm <stage> --recipe <slug> --config <path>`. There is no automatic fallback
when the native runtime or a required schema capability is unavailable.

## Install and run

Start in the Transaction package directory:

```bash
cd packages/experimental/kumo-transaction
uv sync --frozen --extra dev --extra native
uv run --no-sync tfm smoke --recipe blueprint_v1
uv run --no-sync tfm smoke --recipe blueprint_v2
```

Graph and hybrid computation also require `--extra kumo`:

```bash
cd packages/experimental/kumo-transaction
uv sync --frozen --extra dev --extra native --extra kumo
uv run --no-sync tfm smoke --recipe gnn_encoder
uv run --no-sync tfm smoke --recipe gnn_sequence
```

The lockfile pins matching Kumo Engine packages, including the unpublished
`kumo-pytorch` package and DiskGraph runtime, at
`5c3a22cacc43e8de99531693e3b6612f0b34d637`. Install rustup, a C/C++ compiler
and Linux build tools first. A cold native build can take tens of minutes.
Do not install the older published `kumo-engine==1.0.1` bundle over these
packages: it also owns the `diskgraph` namespace.

The standalone [worked loader example](../../examples/kumo-engine/README.md)
also accepts an existing completed index through `--index`. The canonical
recipe CLI owns temporary source indexes for its run and persists replay
anchors and fitted native encoders with its checkpoint.

## What changes

The native path indexes source rows, rather than a prepared copy of every
requested window. Built-in map-valued books receive a bounded scalar Parquet
projection. Custom adapters can supply scalar tables directly through their
`table_source` hook. Keys, relationships, clocks and feature allowlists belong
in that declaration; the recipe chooses the root, history paths and task.
See the [relational input contract](../contracts/relational-inputs.md) and
[shared recipe](../../recipes/relational_inputs/README.md).

Native encoders fit on the selected training anchors. Their fitted vocabulary,
bucket boundaries, field IDs and shared token namespace are persisted and
checked on checkpoint reuse. Native IDs need not match a Python tokenizer's
IDs. **Use a new run directory when changing loaders.** Checkpoints and
preprocessing artifacts from the former prepared-window implementation are
not interchangeable with source-native runs.

A training epoch opens a continuous native iterator, rather than reopening it
for each requested batch index. Native seed and epoch control anchor shuffling.
FutureSubgraph's distinct temporal training stages have one iterator per stage
and epoch, preserving the shared-graph requirement that one batch has a single
cutoff. A producer adapts one owned CPU lookahead batch while the consumer runs
the model. It closes on completion, early stopping or error, before index
resources are released.

The tensor adapter consumes delivered features, masks and clocks directly. It
does not look up Python events or invoke a Python tokenizer during model
batching. Native graph deliveries still pass the graph contracts, including
temporal validation, duplicate rejection, sample alignment and mutation guards.
CPU validation metadata remains separate from the tensors needed for execution.

Undated profile attributes retain their static contract. Dated milestones are
retrieved separately at the prediction cutoff and carry their own ages. They
are independent of transaction history truncation; the native profile stream
keeps the first 64 chronological dated facts. Native flat financial inputs use
the existing fixed scalar feature definitions and native vocabulary delivery.
Their nonoverlapping pretraining groups add one fixed FK per original source
event, rather than copying events for each requested history.

## Preparation and memory tradeoffs

| Concern | Python reference path | Source-native recipe path |
|---|---|---|
| Source ingestion | Constructs an in-process event corpus | Scans and validates file-backed scalar source tables in bounded blocks |
| Feature preparation | Fits the selected Python tokenizer on training histories | Fits native feature encoders on training anchors |
| Sequence training | Prepares histories and tensors, or uses an explicitly prepared token cache | Retrieves and encodes requested source histories per native batch |
| Graph training | Canonical reference runners retain prepared sampled batch lists | Retrieves neighborhoods per batch; does not retain a split of input graphs |
| CPU/GPU overlap | Depends on the caller/cache pipeline | One owned adapted CPU lookahead batch, plus native retrieval buffering |
| Repeated epochs | Can amortize eager preparation and token-cache work | Repeats retrieval and tensor adaptation; retains fitted encoders and index state |

On-demand loading bounds the **derived model-input working set** to active and
prefetched batches instead of every padded history or sampled neighborhood.
This matters when many prediction timestamps reuse overlapping source rows,
especially for deep, wide neighborhoods, and also for repeated sequence
histories. The Python graph sampler can sample one neighborhood at a time too;
it is retention by the caller that determines whether those graphs accumulate.

Both approaches pay preparation costs. Native validation, scalar projection,
key/FK normalization, train-only fitting and index construction take time and
disk space. Python histories, sampled graphs and token caches also cost time,
memory and storage to prepare. Precomputing inputs can amortize repeated work
when the materialized corpus fits the available resources.

This is not an unlimited-memory or performance guarantee. Source indexes,
anchor metadata, the model, optimizer, target vectors, and derived evaluation
arrays still consume resources. FutureSubgraph builds a separate disk-backed
cache of compact future target vectors. Optional hand-engineered reference
baselines make bounded Python history reads **after** model inference; those
reads do not feed model training or tokenization. Embedding and baseline feature
matrices remain proportional to the number of scored samples. Large individual
batches, fanout, depth or prefetch queues can still exhaust memory.

No throughput, preparation-time, memory or quality advantage is claimed merely
from selecting a native backend. Small inputs and repeated short runs can favor
the Python path because native index setup is additional work.

## Supported boundaries

All loader paths preserve explicit label windows and availability. A missing
binary label source is rejected, and non-finite per-event amounts remain
missing values. Source-native training requires a finite amount on an eligible
training event. Historical inputs and future-window targets remain separate.
Legacy timestamped-outcome probes make label-free native history requests and
join their external outcomes after embedding extraction. Their pretraining-cutoff
exclusions and separate task identities remain intact; no input-time label
availability clock is invented for a legacy outcome row.

The pinned native runtime requires whole-second UTC event and cutoff clocks.
A sequence path needs equivalent event and availability clocks. A fixed forward
projection through timed intermediates is supported only when every row and
link along the ownership path was observable by the terminal event time.
Versioned ownership, late-known projections and unsupported reverse/fan-in
intermediate paths fail before training with the affected rule.

Graph hops follow the types reached by the preceding hop, in either FK
direction. Unsupported independent availability/ranking clocks are rejected
when fanout could truncate eligible candidates. Native recent ties follow
index order. A task requiring canonical primary-key ties is refused when that
order cannot be guaranteed at a capped boundary. Explicit graph lookbacks are
accepted only when source coverage proves the lower bound immaterial; otherwise
select the Python reference path. These checks prevent silently changing
ownership, availability or fanout semantics.

The source-native CLI currently supports single-process training. DDP's
single-process eager behavior accepts a native stream; distributed native
padding/loss handling and FSDP streaming are not yet supported. Native packing,
KVT pretraining-window groups, spliced task generation, fine-tuning and surprise
extensions need their existing explicit Python/token-cache workflow. Unsupported
combinations are rejected. A configured token cache keeps its existing cache
builder and fitted tokenizer; it is a separate prepared-input workflow, not
source-native encoding.

## Compare loaders for your workload

Match inputs/anchors, clock and tie rules, model, optimizer, seed, precision,
batch size, optimizer-step count and hardware. Verify sample/label alignment.
Measure source preparation/index time, input-to-first-batch time, steady training
throughput, total wall time and peak host/device memory. Account for target
construction, evaluation and artifact writes consistently. Separate fresh-index
construction from index reuse, synchronize device work at timing boundaries,
and repeat runs. Different early-stopping epoch counts are different amounts of
work.
