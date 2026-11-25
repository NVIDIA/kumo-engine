# Optional native loader integration after 2.0.0a4

`data.loader: python` remains the default for sequence and graph recipes. Native
retrieval is experimental and opt-in; this PR does not establish an end-to-end
performance gain. Select it explicitly in a copied recipe configuration:

```yaml
data:
  loader: kumo_pytorch
```

The data slot selects an input backend, independently of model architecture and
tokenizer. Existing model APIs, sequence compositions, labels and objectives
are unchanged. Native graph inputs pass through the backend-neutral batch
contracts. Future targets remain separate from historical input graphs.

Install with `uv sync --frozen --extra dev --extra native` (add `--extra kumo` for the GNN).
The `native` extra installs the optional external backend; the default Python
loader works without it. The lockfile pins Kumo Engine main at
`5c3a22cacc43e8de99531693e3b6612f0b34d637`, including its experimental
`kumo-pytorch` package and matching DiskGraph runtime. This is a source build:
install Rust through rustup, a C/C++ compiler and the usual Linux build tools
first. A cold native build can take tens of minutes. The published
`kumo-engine==1.0.1` bundle is incompatible with this loader; do not install it
on top of the source packages, because it also owns the `diskgraph` namespace.
The GNN extra now selects the matching `kumo-core` source package.

A configured token cache still selects its existing batch builder. Packing also
retains the existing packer, operating on histories from the selected loader. Neither
setting changes the tokenizer or fits native vocabularies. Run manifests name
the selected history loader; graph manifests additionally identify the native source
revision and separate Python future-target builder. Changing loader changes
resolved-config identity: use a new run directory, and do not reuse an old
managed checkpoint under a different resolved configuration.

Native training collation pads within each minibatch rather than retaining the
previous whole-corpus padding width; valid tokens, masks and labels keep the
same meaning. This can change stochastic training trajectories, so fresh runs
are required for comparisons.

Native sequence selection preserves the inclusive observation start and exact
cutoff, including subsecond events and tied timestamps. Graph sampling has a
narrower schema/clock contract and its own recent-fanout tie ordering; selecting
`python` is required outside that contract. See the
[worked integration guide](../../examples/kumo-engine/README.md#canonical-recipes)
for commands and limitations. These reference integrations still import a
bounded object corpus for validation, tokenization, targets and probes; they do
not establish full-book out-of-core training or native future-aggregate parity.

## Indexed training windows

Native training indexes each declared window as a separate retrieval entity,
so it reads only that window’s events. Repeated source event IDs remain valid,
including distinct events with the same owner/ID and tied timestamps. Event
occurrences retain their original order, labels and timestamps. One physical
training anchor stage serves indexed minibatches through positional sharding;
the final partial minibatch drops only explicitly added selection padding.
This avoids one native storage stage per minibatch and full-owner history reads
for every tiled window. The bounded object corpus and window index still reside
in memory/on local disk; this is not a full-book throughput claim.

## Evidence required before changing the default

Changing the default requires a reproducible comparison of Python and native
loading on representative sequence and graph or hybrid workloads. Native
retrieval alone is insufficient: both paths still import Python objects, fit
tokenizers and collate batches, and graph recipes materialize batches eagerly.
No matched end-to-end results for this integration are included here.

Use identical source records and sample IDs, temporal cutoffs, model, optimizer,
seed, precision, batch sizes, training duration and hardware. Verify valid
inputs, outputs and gradients before timing; use uncapped or otherwise matched
graph neighborhoods so fanout and tie-policy differences do not confound the
comparison. Report all of:

- Input-to-first-batch time, beginning before reading input and including
  validation, index construction/opening, tokenizer fitting and collation.
- Training throughput in effective samples and valid events/tokens per second,
  with the same number of optimizer steps and no padding counted as samples.
- Total wall time from input loading through the same training and artifact
  writing stages; include evaluation in both runs if it is part of the command.
- Peak host resident memory and, when applicable, peak device allocated and
  reserved memory over the entire run, including preparation.

Synchronize device work at measurement boundaries. Publish commands, input
fingerprints, resolved configs, source/dependency revisions, hardware, raw
samples and repeated-run summaries. Separate fresh-index/cold-start runs from
warm-index reuse, and account for preprocessing rather than excluding it from
the native path. A future default switch must cite the reviewed results and
explain the startup, throughput and memory tradeoffs for both workload families.

Full-book streaming and pipeline optimization are follow-up work: they change
ingestion, tokenization and batching together and need their own evaluation.
