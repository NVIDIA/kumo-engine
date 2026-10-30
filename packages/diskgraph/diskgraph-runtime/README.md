# Native batch runtime

`diskgraph-runtime` owns the reusable graph/training-table index and the
worker-sharded iterator that produces `diskgraph_engine::output::RuntimeBatch`.
It has no Python or transport dependency, so native consumers can use the same
batch generation as the Python binding.

`RuntimeIndex::new` opens the parent index and training-table extension once.
`RuntimeIndex::iter` creates per-stage, per-epoch iterators that share the mapped
index and engine. `RuntimeBatchIterator::next_batch` returns a native batch or
`None` at exhaustion. Construction and iteration return typed `RuntimeError`s.

The Python binding retains its constructor arguments, iterator protocol, and
Arrow output conversion. It releases the GIL around native batch generation and
maps runtime errors to the existing DiskGraph Python exception classes.

From `packages/diskgraph`, run:

```bash
cargo test -p diskgraph-runtime
cargo test -p diskgraph-pybind --lib
```
