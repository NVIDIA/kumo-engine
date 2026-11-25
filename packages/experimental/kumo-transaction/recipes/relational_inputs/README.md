# One relational dataset, three model views

This experimental recipe demonstrates direct entity-level binary supervision
from the same adapter declaration and generated anchors. The synthetic dataset
has immutable entity roots, timed intermediate containers, activity rows and a
separate label table. No private schema or dataset is required.

Root-table allowlists become visible profile inputs for sequence models; keys
and clocks remain metadata. Terminal-table allowlists become history events.
Intermediate context features belong to the graph view.

The sequence variant uses the existing profile/event causal decoder. Graph uses
Kumo Engine's GNN, and hybrid uses a selected frozen GNN to condition that same
sequence stack. Graph warm-up and downstream stages use the existing trainer
and validation-loss selection. Raw, embedding and combined raw-plus-embedding logistic probes accompany the
binary task head. Hybrid also trains a sequence-only control with identical
initial sequence weights, binary batches, settings and validation selection;
that control transfers history only. These are mechanics checks, not quality results.

From `packages/experimental/kumo-transaction`:

```bash
uv sync --frozen --extra dev --extra native --extra kumo
uv run --no-sync tfm smoke --recipe relational_inputs
uv run --no-sync tfm smoke --recipe relational_inputs --config graph_smoke
uv run --no-sync tfm smoke --recipe relational_inputs --config hybrid_smoke
uv run --no-sync tfm smoke --recipe relational_inputs --config future_smoke
uv run --no-sync tfm train --recipe relational_inputs --config encoder_pretrain_smoke
```

`config.yaml` is the same small sequence example. `train`, `evaluate` and
`infer` accept the same recipe/config arguments. Set a distinct `--run-dir`
for each variant. Expected artifacts include `model.pt`, its embedded identity
and JSON manifest, split/anchor and tokenizer manifests, embeddings, metrics,
model card and environment provenance. Graph variants also export `encoder.pt`.
A reuse request validates embedded metadata before loading any state.

Python remains the default. To test native retrieval, copy a config, set
`data.loader: kumo_pytorch`, and pass its path with `--config`. The native graph
reference implementation filters row/link availability into cutoff snapshots,
retrieves complete one-hop candidate sets, and applies canonical recent fanout.
Logical fanout is ten per frontier and directed role; physical candidate reads
can be larger. This ensures exact boundary ties, and is not an efficiency claim.
Snapshots and prepared plans close when the command exits.

## Adapting another dataset

Return `AdapterResult(relational=RelationalDataset(...))` from an adapter selected
by `data.adapter`. Declare tables, composite keys, explicit clocks, feature
allowlists, FK roles and label-only tables once. Configure `data.relational.root`,
`history_paths`, and either label timestamps or cutoff-generated anchors. A
Parquet `anchors` table is optional for exact replay and custom predictions in a
new run. Checkpoint reuse requires the identical complete anchor table, including
prediction rows: its digest is part of the recorded experiment identity. Changing
anchors requires a new run and fits the model again; this reference runner does
not support reusing trained weights with a different prediction-anchor table.
See the [input contract](../../docs/contracts/relational-inputs.md) and
[synthetic adapter](../../src/tfm/data/adapters/synthetic_relational.py).

Labels are entity-level outcomes at each anchor, not event fraud labels or the
legacy any-positive truncated-history sequence task. Their horizon, delay and
split fitting cutoffs are explicit. Every path component must be observable at
the input cutoff. Tokenizers fit training histories only. Shared graph batches
split at changes in cutoff/window; evaluation graphs remain disjoint. Sequence
mode allocates no graph and transfers only history inputs.

## Limitations

Inputs and projected histories are bounded in memory. Native sequence projection
can duplicate events across anchors; native graph retrieval deliberately
overfetches candidates for deterministic ties. Graph features currently support
numeric allowlists with presence masks and bias-only identity tables; declared
categorical/text graph inputs are rejected. Versioned ownership is resolved by
point-in-time history traversal, but graph projection rejects it explicitly.
Unsupported rules name their affected path before fitting or training.

The shared runner currently accepts the profile/event decoder, GNN and hybrid,
with single-process eager training. Other sequence compositions retain their
existing event-adapter loading paths. This example does not establish full-book
streaming, native future-target aggregation, quality, or a performance gain.
FutureSubgraph targets remain a separate Python pass with declared source
coverage and a full available horizon; the original graph recipes document
pretrain/downstream encoder workflows. To use a standalone export here, copy
a graph/hybrid config, set `runtime.stage: downstream` and
`runtime.encoder_checkpoint` to its `encoder.pt`. Graph-only downstream runs
train their binary head; hybrid trains its sequence stack with the encoder frozen.
The checkpoint records the imported encoder manifest and byte hash. Retain that
encoder file for managed reuse of downstream runs.

Inference arrays keep label positions aligned with sample IDs. Unknown labels
are NaN with a false `<split>_labels_valid` mask; explicit all-negative label
sources remain valid and use the existing constant-probability probe fallback.
