# Data contract

A **transaction event** is the atomic record: synthetic `entity_id`, `event_id`,
UTC `event_timestamp`, categorical/numeric/text fields, and optional labels.

Labels are not features. Each `EventLabel` has a `level` (`event`, `entity`, or
`sequence`) and `available_at` time. Model inputs must not include fraud,
chargeback, or other post-outcome fields.

A **transaction sequence** is the ordered events for one entity with timestamp
`<= cutoff`, optionally clipped to an observation window, truncated to a maximum
event count. Core-surface sequences are causal.

Optional **entity profiles** (`EntityProfile`: static attributes plus dated
lifelong milestones) and **transfer edges** are domain types for adapters. They
are not consumed by the flat causal decoder.

## External parquet: pragmatiq

`tfm.data.adapters.pragmatiq` maps the dynamiq-ai/pragmatiq four-file layout
(`events.parquet`, `profiles.parquet`, `transfers.parquet`, `labels/*.parquet`)
into these domain types. `user_id` becomes `entity_id`; `event_id` is
synthesized from `(user_id, ts, index)`; `source` is stored on
`categorical["source"]`; string maps are split into numeric/categorical/text.

`events.parquet` is one file or a directory of parquet files. Files are read in
string order of their paths, so `part-10` comes before `part-9`: zero-pad
numbered names, as Spark does. Rows must be sorted by `user_id` across row
groups and files, and by `ts` within each user. A default Spark write is not;
write the book with
`df.repartitionByRange("user_id").sortWithinPartitions("user_id", "ts")`.
The token cache build stops when one user has more events
than `rows_per_block`, and names the value to raise it to. `profiles.parquet`
must be sorted by `user_id` too: each build block reads only its own users'
profiles, and the build refuses an unsorted file.

The 12-field `flat_fields` tokenizer only represents **transaction** events
after `events_for_flat_fields` aliases keys such as `merchant` →
`merchant_name`. App, trading, and communication events are dropped with a
warning. Heterogeneous sources need the key-value-time tokenizer.

Forecast label tables carry `eval_ts`. Recipe `pragma_like` rebuilds
each `default_12m` history with cutoff equal to that row's `eval_ts` so events
after the evaluation point are not encoded. MLM pretraining still uses the
recipe temporal split. Probe train/test is a seeded entity split of labeled
rows (parquet books do not ship predefined folds).

## External parquet: TabFormer

`tfm.data.adapters.tabformer` maps IBM TabFormer columns (`User`, `Amount`,
`Use Chip`, `Merchant Name`, `MCC`, `Is Fraud?`, ...) into `TransactionEvent`.
Resolve the book with `data.path` or `TFM_TABFORMER_ROOT`. Prefer
`temporal_split/*.parquet` when present (`split_protocol: precomputed` is the
v1 80/10/10 cumulative split, documented as an exception to timestamp cutoffs).
Entity subsample (`max_entities`) prefers users with at least one fraud label
on train/test parquet. Sequence-level fraud is **any available positive in the
history window**, not the last event only (split-end last events are almost
never fraud at TabFormer's ~0.1% rate).

Do not store real customer, account, card, or merchant identifiers. Use
synthetic IDs in fixtures and tests.

## Experimental graph samples and future targets

The experimental graph path adds
`GraphAnchor`, explicitly shared/disjoint `GraphBatch`, `GraphHistoryBatch`, and
`FutureSubgraphTargets`. Existing token batch contracts are unchanged. A sample
is identified by sample ID, root entity, cutoff, and observation start; repeated
entities at different cutoffs cannot share a graph neighborhood. Graph/history
sample identities, order, and cutoffs must agree before training.

Table schemas declare numeric feature and target allowlists and forbidden
label/key/clock fields. Temporal rows and links carry availability clocks; only
explicitly immutable tables can be untimed. Node availability, edge orientation,
local indices, duplicate edges, and root ordering are validated.

Future targets are separate numeric values and boolean masks, with ordered
column descriptors, horizon, label delay, availability time, source
observed-through coverage, and target partition boundary. Terminal targets use
`(t,t+H]`, while intermediate label membership uses `t+H`; encoder sampling
remains bounded by `t`. Graph experiment checkpoints bind schema, data/config identity, sampling, preprocessing, splits, and
selected stage state. Existing checkpoint paths are unchanged.


Shared training graphs require identical cutoff and observation-window clocks.
`node_sample` denotes the graph component: anchor index when disjoint, zero for
the single shared component. Shared root indices may repeat; targets and
sample IDs never deduplicate. Validation and embedding/probe inference require
disjoint graphs. Explicit FK directions choose Engine aggregation versus gather;
multi-parent gathers are refused. Sharing changes BatchNorm multiplicities and
is recorded as a training policy, not assumed numerically equivalent.


Graph recipes now default to direct sequence-level binary supervision through
`GraphSupervisedBatch`: one binary label and explicit availability clock per
anchor, with labels outside graph input features. This uses the existing
any-available-fraud-in-truncated-history task. FutureSubgraph remains a separate
opt-in training mode, with its own future targets and fitted statistics.

For Engine hop trimming, `GraphBatch` carries `node_hop`, `num_sampled_nodes`
and `num_sampled_edges` together. Nodes occupy ascending minimum-distance shells
from the batch root set; roots occupy the hop-zero prefix. Incoming edges are
ordered by receiver shell, excluding receivers in the outermost shell. Node
counts have depth+1 entries and edge counts depth entries, including zeroes.
Schemas, totals, root prefixes, inward paths and every layer's index bounds are
validated. Native tensor adaptation requires explicit matching metadata or
`hop_depth` to construct this ordering on CPU. `trim=False` is an explicit
untrimmed control; missing metadata never silently disables requested trimming.


Graph batch constructors validate on CPU. Build or adapt CPU tensors first, then
use `batch.to(device, non_blocking=False)` for a complete value-preserving transfer. The
transfer carries the validation record and cached FK receiver-cardinality facts;
it does not rerun tensor predicates on GPU. Cutoff/window checks index per-anchor
bounds by node/edge owner instead of scanning all rows once per anchor. Engine
compares its required gather roles to the validated facts without GPU sorting.
`FutureSubgraphTargets.valid_count` is also computed on CPU.

For model execution, `batch.for_execution(device, non_blocking=False)` moves
features, edge indices, roots, model history tensors and loss labels/targets.
Ownership, availability/hop tensors, label clocks and raw evaluation features
stay on CPU. This is a mixed-device view, intended for the model call; use
`.to()` when every tensor must be on one device. Graph training, validation and
inference select these views automatically. Other token batches retain their
normal transfer path. No recipe/configuration or checkpoint migration is needed.

CPU construction compiles the transfer layout and flat mutation records once.
Guards avoid recursive field discovery, and transfers avoid reconstructing
validation signatures. An execution view retains its CPU source and checks both
source and copied tensors. Keep both read-only for the view's lifetime, even
after a blocking transfer. Views are ephemeral; CPU sources never cache GPU
views. Immutable anchor layouts and graph-free CPU history projections can be
reused. If a full `.to()` copy is already off CPU, `.for_execution()` first makes
a synchronous complete CPU copy, then selects execution tensors. The normalized
CPU copy is the retained source; the original device batch is not retained and
its later mutations do not affect the copy. This normalization can add transfer
cost; direct views from validated CPU batches remain the fast path.
Serialization reconstructs and validates CPU batches, rebuilding caches.

Validated graph mappings are read-only snapshots. All tensors reachable from a
graph batch are read-only after construction, including history, masks, clocks
and labels. Transfer/model use rejects tracked in-place mutations, tensor
replacement and gradient-flag changes using host-side tensor version records.
For an intentional edit, make new CPU tensors and reconstruct normally (for
example `dataclasses.replace`), which validates again. Same-device `.to()` may
alias tensors; those aliases share the read-only rule. Nonblocking copies require
normal PyTorch stream/lifetime discipline and an unchanged source until complete.

Never write through `.data`, NumPy/shared-memory aliases, foreign kernels, or
other untracked storage access: version counters cannot detect such writes.
External producers must relinquish writes or callers must clone before batch
construction. Concurrent mutation is forbidden. Public graph batch builders and
`.to()` / `.for_execution()` locally use `torch.inference_mode(False)` plus `torch.no_grad()` so they
work inside an outer inference loop while producing version-tracked tensors.
They restore the caller's context; frozen model execution may remain in inference
mode. The private transfer implementation is not a public validation bypass.

```python
with torch.inference_mode():
    batch = sampler.sample(anchors, "users").for_execution(device)
    embeddings = frozen_encoder(batch)
```

Already-created inference input tensors are rejected by low-level constructors:
turning off inference mode does not convert an existing tensor. Make a normal
copy before construction, keeping autograd disabled:

```python
with torch.inference_mode(False), torch.no_grad():
    normal_features = inference_features.clone()
```

Normal caller-supplied tensors remain read-only aliases under the existing rules;
builders do not silently clone every input. CPU validation, mutation detection
and nonblocking-transfer lifetime requirements remain unchanged.


`SequenceSupervisedBatch` contains a `KeyValueTimeBatch`, ordered anchors,
detached binary sequence labels and their availability/cutoff clocks. It has
normal CPU validation, version guards and transfer behavior, with no graph
storage. The binary control consumes it directly:

```python
from tfm.contracts.graph import SequenceSupervisedBatch

control_batch = SequenceSupervisedBatch.from_graph_history(aligned_graph_history_batch)
output = sequence_binary_model(control_batch.to(device))
```

Graph pragmatiq recipes require a nonempty fraud label source and available
source coverage by the cutoff for every supervised owner. Coverage is checked
independently of truncation; the existing any-available-positive history label
is unchanged, including app-only truncated histories and known all-negatives.
Public FutureSubgraph `fit`, `apply` and `loss` reject invalidated target records
before consuming values or allowing gradients.

Graph experiment bundles include the matched binary sequence control and,
for FutureSubgraph full runs, the retained historical next-event control. Reuse reads a
single trusted weight payload and validates its embedded configuration, dataset
and schema metadata before installing state. Sidecar manifests are derived from
the embedded metadata and regenerated on successful reuse. Missing/stale JSON
sidecars cannot change the identity of loaded weights. Source revision/dirty
state may be explicitly unknown without Git; the caller's working directory
never supplies source provenance.


Graph recipe `observed_through` declares source observation coverage; it is not
inferred from the latest event timestamp. A covered interval may have a quiet
tail. Recipes reject empty sources and probe cutoffs beyond that watermark;
FutureSubgraph still requires its full horizon and availability inside declared
coverage. Callers must supply a truthful source watermark.

Transaction graph event joins encode `(entity_id, event_id)` as a compact JSON
array, so separators and Unicode in either identifier cannot collide. The keys
remain metadata rather than model features. Sampling manifests record
`event_key_encoding: json-array-v1`; checkpoint reuse compares this marker as
part of split/sampling identity and rejects incompatible sampling metadata.


Pretrain-only FutureSubgraph runs load no task labels and construct no binary
probe/control batches. Full and downstream stages require available task-label
source coverage for every supervised owner; the truncated-history label rule
is unchanged. A standalone encoder export is a plain state dictionary with a
byte-bound feature/schema manifest. Managed downstream loading verifies it
before installing weights. Named history event arrays remain guarded inputs.


Graph event amounts read the selected `data.amount_field`; the relational schema
keeps canonical amount column names. The source field is part of managed encoder
identity, so matching tensor shapes alone cannot authorize a different amount
definition. Transfer amounts remain the typed `TransferEdge.amount` value.
Pretrain-only graph data has no tokenizer or sequence spec. Binary graph labels
still use the shared cutoff-truncated sequence-label rule; FutureSubgraph builds
only covered future targets and does not require tokenizer-fit histories.


Tokenizer-free graph ingestion still validates the shared event schema. Graph
recipes require at least one finite value for the configured numeric amount
field in training-owner history by a graph-training cutoff, while preserving
per-event missing-value masks.


Shared event validation prohibits label-like keys in categorical, numeric and
text inputs, including text that would otherwise enter a tokenizer vocabulary.
NaN and infinity cannot establish training amount availability; values found
only on held-out owners or in future target windows do not qualify.

Prohibited input-key checks ignore case and surrounding whitespace. This check
does not rename observed fields; their configured identities remain exact.


Raw probe amount aggregates treat an omitted or non-finite event amount as zero.
The mean still divides by the number of events, and the event count and final
calendar features are unchanged. Object and cached aggregation use this same
convention without changing stored amounts or graph missing-value masks.

## Native recipe loading

The opt-in `data.loader: kumo_pytorch` uses the pinned main-source Kumo API.
Python remains the default until matched end-to-end performance is established.
Native anchors bind sample ID, owner and cutoff independently; repeated owners
remain distinct prediction samples. Sequence row IDs resolve only to validated
source records. Exact residual filtering preserves inclusive lower bounds and
microsecond cutoffs before tokenization. Existing train-only tokenizer fits,
label availability and window membership remain authoritative.

The experimental graph sampler requires whole-second clocks, static user roots,
one-parent event/transfer FK roles and event-time availability. It copies native
tensors, validates disjoint ownership and temporal bounds, and then optionally
unions same-cutoff samples for shared training. Native recent-fanout ties use
index order rather than the Python primary-key policy. FutureSubgraph targets
still use a separate Python reference pass. See the
[worked guide](../../examples/kumo-engine/README.md#canonical-recipes) and
[migration notes](../migration/native-loader-default.md).

Resource-backed custom training inputs can implement
`tfm.contracts.resources.Closable.close()` without inheriting a loader class.
The recipe runner closes that resource after fitting or setup failure. Graph
loaders provide `loader_manifest` and `tie_policy` metadata directly; native
resource cleanup is registered when the loader is created.
