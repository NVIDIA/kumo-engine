# Native batch runtime

`diskgraph-runtime` owns the reusable graph/training-table index and the
worker-sharded iterator that produces `diskgraph_engine::output::RuntimeBatch`.
It has no Python or transport dependency, so native consumers can use the same
batch generation as the Python binding.

`RuntimeIndex::new` opens the parent index and training-table extension once.
`RuntimeIndex::iter` creates per-stage, per-epoch iterators that share the mapped
index and engine. `RuntimeBatchIterator::next_batch` returns a native batch or
`None` at exhaustion. Construction and iteration return typed `RuntimeError`s.

`ArrowBatch::try_from(RuntimeBatch)` consumes a native minibatch into Arrow
components and a layout describing its task, graphs and feature banks. It moves
owned vectors into Arrow arrays and shares existing target/feature buffers.
`resolve_reference` selects arrays or record-batch column ranges from that layout;
`validate_layout` checks references received from an external source.

Local sampling and the Flight server use this same conversion. The Flight client
reassembles an `ArrowBatch`, and both Python iterators use one Arrow C Data
exporter. Local iteration performs no IPC or JSON serialization and has no
transport byte limit. The Python binding releases the GIL around sampling and
component construction, then maps runtime errors to the existing DiskGraph
Python exception classes.

From `packages/diskgraph`, run:

```bash
cargo test -p diskgraph-runtime
cargo test -p diskgraph-pybind --lib
```
