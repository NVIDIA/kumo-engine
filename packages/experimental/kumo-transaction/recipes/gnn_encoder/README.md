# Graph encoder with supervised or FutureSubgraph training

**Support tier:** `experimental`.

## Goal and architecture

Learn an entity embedding from its relational neighborhood with direct binary
supervision (default), or without task labels using optional FutureSubgraph. `GNNEncoder` projects numeric node features and directly instantiates Kumo
Engine's `HeteroGNN`, preserving its sum/mean/max aggregation, BatchNorm, ReLU,
and learned skip connections. Linear feature projections enter Engine directly,
without an additional input ReLU. FK directions are declared explicitly so Engine
uses multi-neighbor aggregation and single-parent gather in the correct roles.
It returns one root embedding per anchor and does not instantiate a sequence
model. The [data contract](../../docs/concepts/data-contract.md) defines graph clocks and sample alignment. The
[graph-plus-sequence recipe](../gnn_sequence/README.md) reuses this exact encoder.

The existing optional `kumo` extra supplies the actual Engine implementation;
there is no local substitute when it is missing. The in-memory sampler is the
CPU correctness baseline for graph construction. An adapter accepts
[Kumo Engine](https://github.com/NVIDIA/kumo-engine) PyTorch graph outputs through
`tfm.data.adapters.kumo_graph.from_kumo_graph`; it requires explicit sample/root
bindings, pre-encoded numeric fields, and aligned availability clocks. The
adapter validates native graph tensors; this recipe uses the in-memory sampler.
Native DiskGraph sampling/aggregate parity has not been validated. CUDA model
execution has a short L4 smoke/cost check, not a full training result.

## Task and data assumptions

In the optional pretraining mode, `FutureSubgraph` predicts counts and finite-value amount aggregates over
`(t,t+H]`, with intermediate membership sampled at window end. Only the
historical graph at `t` enters the encoder. Undefined aggregates are masked;
signed-log target selection/normalization fits only self-supervised training
windows. All target horizons must close within declared source coverage and
label availability must lie within the pretraining partition.

All committed configs generate seeded synthetic transactions and transfers.
Training defaults to `data.training_graphs: shared`: per-root sampled
neighborhoods are unioned by canonical node and edge identity. Sharing requires
an identical cutoff and observation window across the minibatch; roots and
targets retain prediction-sample order. Repeated roots remain repeated samples.
Use `disjoint` as an explicit training control. Validation and all embedding/probe
batches always use separate per-sample graphs. Sharing changes node multiplicity
and BatchNorm statistics, so it is not a promise of identical training outputs.
The graph contains immutable user identity roots (a constant numeric bias),
events, and transfer rows with sender/receiver FK roles. Only amount and its
presence/log representation enter graph features/targets; IDs, outcomes, and
clocks do not. Event/transfer observations are assumed available at event time.
Histories include multiple channels when the source supplies them. Static
profile attributes are not used in these first graph recipes.

The stock configs train the GNN directly using the available binary labels
at the probe cutoff, selecting on separate validation owners. No future targets,
future normalization, or historical SSL windows are required. The sequence-only
control trains on the same labels and owner splits without any GNN modules.

Optional FutureSubgraph pretraining uses three historical anchors on training owners. A separate owner
subset and later historical anchor select the checkpoint by valid-entry-weighted
self-supervised loss. The historical boundary and all sample mappings are saved.
In that mode the sequence baseline uses next-event training and its tokenizer
fits only the earliest historical training anchor. In supervised mode the
tokenizer fits only training owners at the task cutoff.

Downstream task: the existing **any available fraud in the truncated history**
sequence label, not future-fraud prediction. Probe train/validation/test owners
are disjoint at the configured later cutoff. Cross-owner graph context through
transfers is permitted and explicitly recorded. This is not an entity-isolated
graph experiment. Pretraining is unspliced; no identity-theft splice is enabled.

## Commands

From the repository root, after `uv sync --extra dev --extra kumo`:

```bash
uv run tfm smoke --recipe gnn_encoder
uv run tfm smoke --recipe gnn_encoder --config pretrain_smoke
uv run tfm train --recipe gnn_encoder --config pretrain --run-dir outputs/gnn_encoder/pretrain
uv run tfm train --recipe gnn_encoder --run-dir outputs/gnn_encoder/example
uv run tfm infer --recipe gnn_encoder --run-dir outputs/gnn_encoder/example --reuse-checkpoint
uv run tfm evaluate --recipe gnn_encoder --run-dir outputs/gnn_encoder/example --reuse-checkpoint
uv run tfm report --recipe gnn_encoder --run-dir outputs/gnn_encoder/example --reuse-checkpoint
```

`config.smoke.yaml` runs three optimizer steps per trained stage on 24 generated
owners. `config.yaml` uses 64 owners and 30 steps. Both are CPU-safe wiring
examples, not benchmark settings. Eager training is supported; DDP/FSDP and
packing are refused. Use `--device cuda` only with a compatible PyTorch runtime;
GPU validation is separate from the required CPU smoke. A short GPU timing
comparison is defined in the [three-arm benchmark](../../benchmarks/ablations/graph_backends/README.md).

To read a local pragmatiq book, copy a config and set `data.adapter: pragmatiq`,
`data.path` (or `TFM_PRAGMATIQ_ROOT`), `n_entities`, `pretrain_end`,
`probe_cutoff`, `observed_through`, and `horizon_days` consistently with that
book. Coverage is a caller declaration, not inferred from an absence of rows.
FutureSubgraph requires complete forecast-window coverage. Full runs also need
training-owner history by the earliest tokenizer-fit cutoff; pretrain-only runs
have no tokenizer-fit history requirement. Direct supervised mode needs history
and available labels at the task cutoff. The loader selects owners without fraud stratification and
builds the induced graph, omitting transfers to owners outside the sample.
Existing [dataset roles](../../docs/reference/datasets.md) still apply; never
combine synthetic-fixture and pragmatiq results. The in-memory graph path is
intended for small samples.

## Minimal composition

```python
from tfm.experimental import GNNEncoder, GNNEncoderModel, FutureSubgraph

spec = GNNEncoder(
    node_widths=(("users", 1), ("events", 2)),
    relations=(("users", "events", "events"), ("events", "rev_events", "users")),
    root_type="users",
    foreign_key_relations=(("events", "rev_events", "users"),),
    hidden_size=32,
    depth=2,
)
from tfm import Binary

model = GNNEncoderModel(spec, objective=Binary())
# model(graph_supervised_batch) trains directly on available binary labels.
# Alternatively, with fitted and aligned future-target columns:
pretrain_model = GNNEncoderModel(
    spec, target_columns=("future_amount",), objective=FutureSubgraph()
)
# pretrain_model(aligned_graph_training_batch) trains the future objective.
# model(graph_batch, compute_loss=False)
# returns ModelOutput.embeddings [anchors, 32]. Target columns must match the
# fitted FutureTargetTransform exactly; raw targets do not enter GraphEncoder.
```

`model.encoder` is the head-free `GraphEncoder` reused by the hybrid.
`GraphSampler.sample` builds validated disjoint batches by default, including
repeated entities with distinct sample IDs/cutoffs. `sharing="shared"` explicitly
unions same-cutoff neighborhoods for training; mixed clocks/windows are refused. The graph schema and
allowlists are explicit in `tfm.data.graph`; label construction is outside
model code. Public contracts and components document their shapes and semantics in docstrings.

## Baselines, outputs, and limitations

The evaluation reports raw five-aggregate logistic, embedding-only, combined,
sequence-only matched-mode embedding/combined, and graph-tabular logistic arms.
Graph-tabular features count nodes and sum their encoded numeric features on
the same sampled input graphs. The sequence model uses the same supervised labels or historical
pretraining anchors, depending on the selected mode, and the same probe rows.
`task_head` also evaluates the trained binary head in supervised mode. These logistic arms exercise the wiring;
no XGBoost headline or quality lift is claimed for this recipe.

Outputs include `resolved_config.yaml`, validation/split/tokenizer/checkpoint/
embedding/probe/environment manifests, `model.pt`, `model.manifest.json`,
`train_log.json`, `embeddings.npz`, `metrics.json`, `metrics.md`, and
`model_card.md` (report/smoke). Embedding arrays include sample and entity IDs
and labels. Checkpoints retain target descriptors/statistics, fitted tokenizer,
data/config fingerprints, graph schema, validation selections, and source
revision, Engine version, and sharing policy. `--reuse-checkpoint` verifies config and data identities before loading
weights. It is inference reuse, not exact optimizer/cursor training resume.

Only explicitly allowlisted numeric targets are supported. Six-hop settings
are configurable through `depth` and `target_depth` but are not a quality or
memory guarantee. The reference path materializes sampled graphs and targets
in memory, and uncapped terminal reductions can be costly. No private data,
pretrained weights, production readiness, paper fidelity, or performance claim
is supplied. Full-book and native-loader validation remain future work; a short CUDA smoke
and cost check does not validate a full training campaign.


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
uv run tfm smoke --recipe gnn_encoder --config downstream_smoke --run-dir outputs/gnn_encoder/downstream
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
