# GNN plus sequence model

**Support tier:** `experimental`.

## Goal and architecture

Condition Blueprint v2's ordered history on a learned relational embedding.
`GNNSequence` contains exactly the graph encoder from
[`gnn_encoder`](../gnn_encoder/README.md), a trainable projection and graph-field
token, and Blueprint v2's profile/event encoders and causal history decoder:

```text
graph at cutoff -> frozen GNN -> projection + graph field -> append to profile tokens
history tokens -> event encoder -> event summaries
profile encoder + event summaries -> causal history decoder -> last valid slot
```

SDPA is the default attention kernel; `model.composition.attention: eager`
is the correctness control. The graph token enters **before** profile encoding. It is not concatenated after
sequence pooling. The sequence stack is initialized afresh. The shared GNN
is head-free and stays frozen, including buffers and evaluation mode.
The [data contract](../../docs/concepts/data-contract.md) defines graph/history sample alignment and future-target availability.

## Tasks and stages

1. Train the standalone GNN directly on available binary task labels (default),
   or select `training.objective: future_subgraph` for historical self-supervision.
   Restore the checkpoint with lowest matching validation loss, including buffers.
   In FutureSubgraph mode downstream fraud labels do not enter this stage.
2. Discard the temporary aggregate head and optimizer. Freeze the GNN and train
   a fresh graph projection, graph field, profile/event/history stack, and binary
   head. Select by binary validation loss on separate probe validation owners.
   Each stage has its own optimizer; moments persist within a stage's epochs.
   Training graphs share same-cutoff neighborhoods by default. This applies to
   the graph pretraining stage and frozen-GNN computation in the sequence stage.
   Selection, validation/test, and embedding extraction remain disjoint.

The supervised target is **any available fraud in the truncated history**, a
sequence-level detection task with existing label-availability semantics.
A graph sampled at the final cutoff can contain information after an earlier
history position, so within-history next-event objectives and surprise are
refused. A causal history mask does not make that conditioning prefix-safe.
FutureSubgraph initially trains the graph stage only; joint GNN/sequence
self-supervision is not included. Supervised GNN warm-up is the default.

## Data assumptions and commands

Use the same generated transactions/transfers, owner splits, observed-through
boundary, target masks, train-only tokenizer/normalization, local-pragmatiq
configuration, and cross-owner graph access policy as the
[graph-only recipe](../gnn_encoder/README.md#task-and-data-assumptions).
The default and smoke configs use binary supervision in both recipes.
The `pretrain` and `pretrain_smoke` variants select FutureSubgraph for the GNN
stage while the hybrid objective remains binary. Both recipes include the same sequence-only baseline.
No task labels or profile attributes become graph input features.

```bash
uv sync --extra dev --extra kumo
uv run tfm smoke --recipe gnn_sequence
uv run tfm smoke --recipe gnn_sequence --config pretrain_smoke
uv run tfm train --recipe gnn_sequence --config pretrain --run-dir outputs/gnn_sequence/pretrain
uv run tfm train --recipe gnn_sequence --run-dir outputs/gnn_sequence/example
uv run tfm infer --recipe gnn_sequence --run-dir outputs/gnn_sequence/example --reuse-checkpoint
uv run tfm evaluate --recipe gnn_sequence --run-dir outputs/gnn_sequence/example --reuse-checkpoint
uv run tfm report --recipe gnn_sequence --run-dir outputs/gnn_sequence/example --reuse-checkpoint
```

The smoke config has an explicit seed, 24 owners, and three steps per stage;
the default has 64 owners and 30 steps. Training is single-process eager;
CPU is the checked baseline and GPU execution needs separate validation.
These are runnable educational examples, not a recommended training budget.

## Minimal composition

```python
from tfm.experimental import GNNEncoder, GNNSequence, GraphEncoder, GNNSequenceModel
from tfm.experimental import ProfileEventDecoder

spec = GNNEncoder(
    node_widths=(("users", 1), ("events", 2)),
    relations=(("users", "events", "events"), ("events", "rev_events", "users")),
    root_type="users",
    foreign_key_relations=(("events", "rev_events", "users"),),
    hidden_size=32,
    depth=2,
)
encoder = GraphEncoder(spec)  # normally restore the selected pretrained encoder
sequence = ProfileEventDecoder(hidden_size=32, num_heads=4, vocab_size=512)
model = GNNSequenceModel(GNNSequence(spec, sequence), encoder)
# model(aligned_graph_history_batch) returns cutoff-level binary loss.
# GraphHistoryBatch validates sample IDs, entities, cutoffs and label availability.
```

## Baselines, outputs, and limitations

The same graph-only raw/embedding/combined, sequence-only, and graph-tabular
logistic comparisons run on identical scored rows. Additionally,
`graph_only_embedding` and `graph_only_combined` probe the exact frozen GNN
inside this hybrid, isolating what the sequence stage adds. The raw arm keeps
the same five history aggregates; graph-derived baselines remain separate.
`task_head` evaluates the trained hybrid binary head. No XGBoost lift or future-fraud quality claim is made.

Artifacts and checkpoint reuse follow the
[graph-only recipe](../gnn_encoder/README.md#baselines-outputs-and-limitations).
The checkpoint includes the entire frozen encoder and selected trainable
sequence state, as well as the independent sequence baseline. Inference reuse
verifies config/data identity; exact training continuation is not supported.
The graph sampler/target builder materialize small samples in memory. The actual Engine GNN supplies message passing in both stages. Native
Kumo graph tensor adaptation is available, while native sampling/aggregate
parity, full-book scaling remain unverified; a short L4 CUDA smoke/cost check does not
validate a full training campaign.


## Engine execution contract

`model.composition.granularity.trim: true` is the default. The sampler groups
nodes by minimum hop distance and incoming edges by receiver hop, validates
prefix bounds, and passes per-type node and per-role edge counts to Engine.
This works with shared training graphs and disjoint validation/test graphs.
Native tensor callers must supply matching metadata or explicitly request
`from_kumo_graph(..., hop_depth=depth)` to construct it on CPU. Missing counts
fail when trimming is requested. `trim: false` provides an untrimmed control.
The feature projection has no additional input ReLU; Engine retains its own
internal activations. The sequence stack defaults to SDPA and supports eager
attention with matching masks. The historical L4 cost record predates these
changes; it is not a measurement of current trimmed-model throughput.

Direct supervised training currently supports the existing sequence-level
binary task only. Regression and joint end-to-end GNN/sequence training are
not provided. See the [batch contract](../../docs/concepts/data-contract.md).


Graph/label/target batches validate on CPU before transfer. `.to(device)` carries
validation without repeating GPU tensor checks. Treat all batch tensors as
read-only; use fresh CPU tensors and normal reconstruction for intentional
changes. Tracked mutations are rejected before transfer/model use. External or
`.data` writes are forbidden; see the linked batch contract for ownership,
nonblocking transfer and inference-mode rules. Public builders and transfers
work inside an outer inference loop using local version-tracked no-grad
construction; already-created inference input tensors require explicit copies.
Training, validation and inference select
`.for_execution(device)` views: model inputs/targets move to the device while
validation metadata stays on CPU. Full `.to(device)` copies remain supported.
Both CPU source and device tensors remain read-only while a view is in use;
flat mutation records check them without rebuilding recursive signatures.


Checkpoint reuse reads embedded metadata from `model.pt` before installing any
weights. It validates the requested config, data and schemas against that
metadata and regenerates JSON sidecars from it. Stale/missing sidecars cannot
supply model identity. Source provenance is resolved against the source checkout; installed/no-Git
builds record an unknown revision instead of failing after training.

The sequence-only **binary** control trains and selects on exactly the hybrid's
history/label batches, with the same initial shared-stack parameters, seed and
training settings. Its `SequenceSupervisedBatch` has no graph tensors, so neither
training nor evaluation transfers unused graphs. `sequence_supervised_embedding`,
`sequence_supervised_combined` and `sequence_supervised_task_head` name these
comparisons. FutureSubgraph mode additionally retains the historical next-event
control under `sequence_only_*`; in direct binary mode those legacy names refer
to the binary sequence control. Encoding is shared with Blueprint v2.

Full and downstream pragmatiq runs require available fraud-label source coverage
for every supervised owner. FutureSubgraph pretrain-only runs neither load task
labels nor build binary controls/probes. Binary pretraining still requires labels.
Valid all-negative sources and app-only truncated histories retain their existing
label semantics. No owner is selected by label value.


Set `data.observed_through` to the verified source observation watermark. The
last event may be earlier than that watermark; quiet tails are valid. Event join
keys use JSON-encoded identity pairs. Sampling metadata records `event_key_encoding: json-array-v1` as part of the sample identity.


## Separate pretraining and downstream runs

All variants use `tfm.config.RecipeConfig`. `runtime.stage` is `full` by default.
The pretrain-only variant trains just the graph stage and supports unlabeled
books. From the repository root:

```bash
uv run tfm train --recipe gnn_encoder --config pretrain_only_smoke --run-dir outputs/gnn_encoder/pretrain_only
uv run tfm smoke --recipe gnn_sequence --config downstream_smoke --run-dir outputs/gnn_sequence/downstream
```

The downstream variant points `runtime.encoder_checkpoint` at that plain encoder
artifact. Hybrid training freezes it; graph-only downstream training fine-tunes
it with binary labels. Downstream builds no future targets. `encoder.pt` is a
plain state dictionary and `encoder.manifest.json` binds its bytes to feature,
graph/module schema and Engine provenance. Managed loading verifies the manifest
before installing weights; a custom model can also load the state directly:

```python
import torch
from tfm.experimental import GraphEncoder

encoder = GraphEncoder(spec)  # the same graph feature/schema contract
encoder.load_state_dict(torch.load("encoder.pt", map_location="cpu", weights_only=True))
```

Pretrain-only outputs are weights, encoder metadata, split/config/environment
manifests and the graph-stage log; they contain no binary metrics. Scoring uses
a downstream/full stage with labels. The historical next-event control is kept
in full FutureSubgraph runs; the matched binary control remains aligned with
the hybrid in full/downstream runs. See the [extension guide](../../docs/reference/extending.md#graph-composition-extension-points)
for custom architectures, context histories, readouts and task heads.

### Shared feature and position settings

`data.amount_field` (default `amount`) selects the observed numeric event field
used by graph features, FutureSubgraph amount aggregates, and raw probe features.
Canonical graph column names stay `amount`, `amount_log`, and `present`;
transfer amounts still come from `TransferEdge.amount`. The encoder manifest
records the source field, and managed loading rejects a different selection.
That field must have at least one finite value in training-owner history by a
graph-training cutoff; individual missing or non-finite graph amounts keep their
existing masks. Every stage validates the shared event schema,
including tokenizer-free pretraining, and rejects label-like input fields.

`model.composition.position` defaults explicitly to `time_rope` in graph configs.
Set `nope` to disable elapsed-time rotary positions in the hybrid and both
sequence controls. Calendar and within-event position features remain as
configured. The causal attention mask is unchanged.

A `runtime.stage: pretrain` run builds no sequence batches and does not resolve,
fit, or persist a tokenizer. FutureSubgraph needs covered forecast windows but
no history at the sequence tokenizer's earliest fit cutoff. Binary graph-only
pretraining still derives labels from cutoff-truncated histories using the
shared sequence-label rule. Full and downstream runs fit or restore a tokenizer
for their sequence arms. Graph recipes reject `data.token_cache`; they currently
load adapter events in memory.

## Loader selection

Kumo Engine's experimental PyTorch/DiskGraph loader is the default
(`data.loader: kumo_pytorch`). Install `--extra native` for sequences and
`--extra kumo` for graph models. The [worked loader guide](../../examples/kumo-engine/README.md#canonical-recipes)
provides setup, commands and temporal/schema limits. Set `data.loader: python`
in a copied configuration to use the reference loader. Both retain preparation
costs; see [memory and materialization tradeoffs](../../docs/migration/native-loader-default.md#materialization-memory-and-preparation).
The current runners still ingest bounded object corpora, so this default does
not establish out-of-core training or a universal speed/memory advantage.
