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
`SeedSchedule::from_json` constructs the same scheduler from seed-side metadata
and sidecars without opening graph adjacency or feature files. Construction,
selection, and sampling return typed `RuntimeError`s.

The Python binding retains its constructor arguments, iterator protocol, and
Arrow output conversion. It releases the GIL around native batch generation and
maps runtime errors to the existing DiskGraph Python exception classes.

The low-level Python schedule is available as `diskgraph._c.SeedSchedule`.
Its constructor accepts JSON with these fields:

| Field | Meaning |
| --- | --- |
| `stage_key` | Stage or RHS node-type name used for the shuffle seed. |
| `ranges` | Disjoint inclusive `[min_id, max_id]` ranges, in traversal order. |
| `num_workers`, `worker_id` | Worker count and zero-based shard identity. |
| `shuffle`, `rnd_seed`, `epoch` | Shuffle settings; shuffling requires a seed. |
| `batch_size`, `drop_last` | Positive batch bound and partial-tail policy. |
| `rhs` | Whether the ranges describe one dense RHS candidate domain. |
| `selected_stage_node_ids` | Optional ordered selection, preserving duplicates. |
| `weighted_seed_sampling_json` | Optional weighted-sampling settings and sidecars. |
| `bucket_extension_root` | Optional extension containing timestamp buckets. |

`next_seed_ids()` returns canonical IDs or `None` at exhaustion. An explicit
`sample_request(ids)` accepts a nonempty batch within the configured size and
stage/RHS domain, including after local iteration is exhausted. Invalid requests
raise the existing configuration/data exceptions and leave the cursor unchanged.
The Python result includes configured neighborhoods and history, matching
`next_batch()` for the same seeds. Existing length-planned iteration retains its
planner and cache behavior.

Direct Cargo tests need the platform feature flags used by the wheel build.
From `packages/diskgraph`, run on x86_64:

```bash
RUSTFLAGS="-C target-feature=+aes,+sse2" cargo test -p diskgraph-runtime
RUSTFLAGS="-C target-feature=+aes,+sse2" cargo test -p diskgraph-pybind --lib
```
