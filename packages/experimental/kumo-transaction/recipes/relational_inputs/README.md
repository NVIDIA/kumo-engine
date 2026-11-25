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

Kumo Engine is the default (`data.loader: kumo_pytorch`). To use the Python
reference path, copy a config, set `data.loader: python`, and pass its path with
`--config`. See [preparation and memory tradeoffs](../../docs/migration/native-loader-default.md#preparation-and-memory-tradeoffs).
The native runner indexes source tables once, fits native feature encoders on
training anchors and streams owned model-ready tensors. It keeps one adapted
CPU lookahead batch and no split-wide input graph list. Native seed and epoch
control sample ordering. Resources close after producers finish or stop early.

## Adapting another dataset

Register a `table_source` factory on an adapter selected by `data.adapter`.
Return `TableSource(tables=..., relationships=..., files=...)`: scalar Parquet
files plus one declaration of tables, keys, clocks, feature allowlists, FK roles
and label-only tables. The source hook does not construct Python event objects.
The same schema can also back a Python adapter returning
`AdapterResult(relational=RelationalDataset(...))` for the reference path. Configure `data.relational.root`,
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

Native ingestion scans scalar source tables in bounded blocks; native model
inputs are retrieved and adapted per batch. Fixed projections through timed
intermediates preserve one terminal row per source path, not one copy per
anchor. The pinned backend requires whole-second UTC clocks, equivalent event
and availability clocks for histories, and fixed ownership observable by event
time for projected paths. Multiple paths to the same history terminal,
versioned ownership and unsupported timed fan-in/reverse projections are
rejected. The Python reference retains the broader data-contract traversal.

Graph features support numeric allowlists with missing-value masks. Native
fanout is applied at each reached table type and directed FK role. Independent
ranking/availability clocks, meaningful per-hop lookbacks and canonical capped
ties are refused when the backend cannot preserve the requested rule. The
[migration guide](../../docs/migration/native-loader-default.md#supported-boundaries)
describes these fail-before-training checks. A six-hop/fanout-ten configuration
does not imply support for every possible schema or clock policy.

The shared runner currently accepts the profile/event decoder, GNN and hybrid,
with single-process eager training. Stock transaction sequence recipes also have source-native CLI streams.
This example does not establish native future-target aggregation, quality, or
a measured performance gain. FutureSubgraph targets use a separate disk-backed
Python reference traversal with declared source
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
