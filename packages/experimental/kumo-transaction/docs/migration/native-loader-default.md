# Kumo Engine as the recipe loader default

Sequence, graph and hybrid recipes default to Kumo Engine's experimental
PyTorch/DiskGraph loader, `data.loader: kumo_pytorch`. This is an integration
choice, not a claim that it is faster or uses less memory for every workload.
The Python reference loader remains available:

```yaml
data:
  loader: python
```

Copy the recipe configuration, set that field, and pass its path with
`tfm <stage> --recipe <slug> --config <path>`. Loader selection is independent
of model architecture and tokenizer. Existing direct library history helpers
keep their Python defaults; recipe runners pass the resolved setting explicitly.
There is no automatic fallback if native dependencies or schema capabilities
are unavailable.

## Install and migrate

From `packages/experimental/kumo-transaction`, install the native loader for
sequence recipes:

```bash
uv sync --frozen --extra dev --extra native
uv run --no-sync tfm smoke --recipe blueprint_v1
```

Graph and hybrid models also need `--extra kumo`. The lockfile pins matching
Kumo Engine source packages at
`5c3a22cacc43e8de99531693e3b6612f0b34d637`, including the unpublished
`kumo-pytorch` package and DiskGraph runtime. Install rustup, a C/C++ compiler
and Linux build tools first. A cold native build can take tens of minutes.
Do not install the older published `kumo-engine==1.0.1` bundle over these
packages: it also owns the `diskgraph` namespace. The GNN uses the matching
`kumo-core` source package.

Sequence runs explicitly selecting Python need no native extra. Their
preparation is performed by the runner; users do not need to hand-build
sequences. Keep `data.loader: python` in a copied configuration to reproduce
the prior loader choice. Changing the resolved loader changes checkpoint
identity: use a new run directory rather than reusing a managed checkpoint
under a different configuration.

A configured token cache retains its existing batch builder. Packing retains
the existing packer over histories from the selected loader. Neither setting
changes the tokenizer or fits native vocabularies. Native training collation
pads within each minibatch rather than retaining a whole-corpus padding width;
valid tokens, masks and labels retain their meaning. Fresh training trajectories
need not be bitwise identical across loaders.

## Materialization, memory and preparation

A label or prediction anchor identifies a sample at a particular cutoff; label
values never become input features. Many anchors for the same entity can have
overlapping histories or neighborhoods. Materializing all of those inputs can
duplicate substantial data even when the underlying source tables are modest.

| Concern | Python reference path | Kumo Engine path |
|---|---|---|
| Retrieval | Select histories and sample neighborhoods with in-process reference code | Retrieve historical rows or graph candidates through a native index |
| Preparation | Validate source data, fit train-only encoders and construct reference batches/caches | Validate data, fit train-only encoders and additionally build/open native indexes |
| Repeated sequence use | Eager sequence recipes can reuse already-collated tensors; explicit token caches have their own preparation/storage costs | Native sequence training retrieves and collates each indexed minibatch; repeated tokenization can add CPU work |
| Graphs | Samples neighborhoods on demand, but canonical graph runners retain prepared batch lists | Uses native sampling, but those runners still retain prepared graph batches; switching the backend alone does not remove them |
| Small inputs | Avoids native build/index setup | Native setup and conversion overhead can dominate |

On-demand materialization can bound the **derived input working set** to active
and prefetched batches instead of retaining every padded sequence or sampled
subgraph. This can be valuable with many label timestamps, long histories,
multiple paths, or deep/high-fanout neighborhoods. The Python graph sampler
also constructs individual neighborhoods on demand; this technique is not
exclusive to a particular backend.

Both paths still pay for source ingestion, validation, feature fitting and
indexes or caches. Precomputing more inputs spends time, memory and storage
up front and can amortize that work over repeated epochs. Computing inputs
per minibatch reduces retention only when callers release them, and can repeat
retrieval, tokenization and collation work each epoch. Large fanout/depth or
aggressive prefetch can exhaust memory even with on-demand loading; bound batch
size, history length, neighborhoods and retained caches.

**Current limit:** these reference runners import a bounded Python object
corpus for validation, tokenization, targets and probes. Native indexed windows
can duplicate event occurrences across anchors, and graph runners still eagerly
retain batches. A disk-backed retrieval backend alone does not make this pipeline
out of core. Do not infer unlimited dataset capacity or a measured memory,
preparation-time or throughput advantage from the default selection. Full-book
streaming requires a separate ingestion/encoding/batching integration.

## Temporal and backend compatibility

Native anchors bind sample ID, owner and cutoff independently; repeated owners
remain distinct samples. Sequence selection preserves inclusive observation
starts, microsecond cutoffs and tied-event order before the existing tokenizer.
Train-only fitting, label availability, masks and explicit window membership
remain authoritative.

Native training indexes each declared window independently and reads its exact
members, including repeated event IDs and tied timestamps. One physical anchor
stage serves indexed minibatches through positional sharding; the final batch
discards only explicitly added selection padding. This window index is still
bounded and does not eliminate source/window preparation.

The legacy event-adapter native graph sampler requires static user roots,
whole-second event/cutoff clocks, one-parent user-to-event FK roles,
event-time availability and no explicit observation-start bound. Native recent
fanout ties follow index order, which can differ from Python primary-key order.
Select Python explicitly for sources outside that native contract. Native
inputs pass through the backend-neutral graph validation and ownership checks.
FutureSubgraph targets remain a separate Python reference traversal and never
enter historical inputs.

The [shared relational recipe](../../recipes/relational_inputs/README.md)
provides one table/key/relationship/clock/allowlist declaration for sequence,
GNN and hybrid, with generated or supplied anchors. Its fixed-FK native graph
path supports timed intermediates, both FK directions, exact cutoff snapshots
and canonical fanout ties by retrieving complete eligible candidate sets before
capping. Physical reads can therefore exceed logical fanout. Numeric graph
features are supported; versioned graph topology and categorical/text graph
features fail explicitly. Sequence histories support point-in-time ownership
traversal. See the [input contract](../contracts/relational-inputs.md) and
[worked loader guide](../../examples/kumo-engine/README.md#canonical-recipes).

## Comparing loaders for a workload

Compare identical inputs/anchors, clocks, model, optimizer, seed, precision,
batch sizes, optimizer-step counts and hardware. Verify temporal correctness
and input/output alignment; account for different graph fanout tie policies.
Report preparation/index time, input-to-first-batch time, steady training
throughput, total wall time and peak host/device memory. Include preprocessing,
collation, targets, evaluation and artifact writes consistently. Separate
fresh-index construction from index reuse, synchronize device work at timing
boundaries and repeat runs. Different early-stopping epoch counts are different
amounts of work. Select the backend and cache strategy that fit the workload.
