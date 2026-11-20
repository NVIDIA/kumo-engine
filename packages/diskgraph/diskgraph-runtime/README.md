# Native batch runtime

`diskgraph-runtime` owns the reusable graph/training-table index and the
worker-sharded iterator that produces `diskgraph_engine::output::RuntimeBatch`.
It has no Python or transport dependency, so native consumers can use the same
batch generation as the Python binding.

`RuntimeIndex::new` opens the parent index and training-table extension once.
`RuntimeIndex::iter` creates per-stage, per-epoch iterators that share the mapped
index and engine. `RuntimeBatchIterator::next_batch` returns a native batch or
`None` at exhaustion. `next_seed_ids` separates logical selection from sampling;
`sample_request` samples explicit canonical stage/RHS IDs without advancing that
cursor. Local iteration and explicit requests use the same sampling operation.
`SeedSchedule::new(SeedScheduleConfig)` constructs the same scheduler from typed
seed-side metadata and sidecars without opening graph adjacency or feature files. Construction,
selection, and sampling return typed `RuntimeError`s.

The Python binding retains its constructor arguments, iterator protocol, and
Arrow output conversion. It releases the GIL around native batch generation and
maps runtime errors to the existing DiskGraph Python exception classes.

Local iteration and standalone scheduling share this typed initializer and its
validation, including checking all selected IDs before worker sharding. There is
no JSON schedule constructor.

The low-level Python schedule is available as `diskgraph._c.SeedSchedule` and
accepts a typed `diskgraph.SeedScheduleConfig`:

```python
from diskgraph import SeedScheduleConfig
from diskgraph._c import SeedSchedule

schedule = SeedSchedule(SeedScheduleConfig(
    stage_key="train_seeds",
    ranges=[(0, 99)],
    batch_size=32,
    shuffle=True,
    rnd_seed=42,
))
seed_ids = schedule.next_seed_ids()
```

The Rust configuration and Python dataclass have these fields. Python defaults
to one worker, worker zero, epoch zero, no shuffle, no tail dropping, and no
optional mode:

| Field | Meaning |
| --- | --- |
| `stage_key` | Exact sampling-mode key used for the shuffle seed; see below. |
| `ranges` | Disjoint inclusive `[min_id, max_id]` ranges, in traversal order. |
| `num_workers`, `worker_id` | Worker count and zero-based shard identity. |
| `shuffle`, `rnd_seed`, `epoch` | Shuffle settings; shuffling requires a seed. |
| `batch_size`, `drop_last` | Positive batch bound and mode-specific partial-tail policy; see below. |
| `rhs` | Whether the ranges describe one nonempty dense RHS candidate domain. |
| `selected_stage_node_ids` | Optional ordered selection, preserving duplicates. |
| `weighted_seed_sampling_json` | Optional weighted-sampling settings and sidecars. |
| `bucket_extension_root` | Optional extension containing timestamp buckets. |

To reproduce local ordering, flat schedules must use the exact `stage_table`
argument passed to the native iterator, such as `train_seeds`, rather than the
Python stage name `train`. Weighted and timestamp-grouped schedules use the
namespaced catalog key, such as `<namespace>/train_seeds`. RHS schedules use the
RHS node-type name. These keys are part of the random seed calculation.

Flat and RHS schedules honor `drop_last`. Timestamp-grouped schedules retain
each bucket's partial tail even when `drop_last=true`. Selected and weighted
schedules reject `drop_last=true`. RHS ranges must contain exactly one inclusive
range with `min_id <= max_id`; a single-node range is valid, and reversed ranges
are rejected rather than interpreted as an empty candidate domain. Grouped
sidecar IDs outside the supplied stage ranges raise a data error when selected.

`next_seed_ids()` returns canonical IDs or `None` at exhaustion. An explicit
`sample_request(ids)` accepts a nonempty batch within the configured size and
stage/RHS domain, including after local iteration is exhausted. Invalid requests
raise the existing configuration/data exceptions and leave the cursor unchanged.
The Python result uses the same output fields and configured neighborhoods and
history as `next_batch()`. Explicit requests do not make randomized sampling
deterministic: for example, negative sampling with `neg_rnd_seed=None` draws
fresh random negatives, so repeated requests for the same IDs can differ.
Existing length-planned iteration retains its planner and cache behavior.

Direct Cargo tests need the platform feature flags used by the wheel build.
From `packages/diskgraph`, run on x86_64:

```bash
RUSTFLAGS="-C target-feature=+aes,+sse2" cargo test -p diskgraph-runtime
RUSTFLAGS="-C target-feature=+aes,+sse2" cargo test -p diskgraph-pybind --lib
```
