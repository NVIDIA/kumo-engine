# Native Flight gateway

`diskgraph-gateway` serves complete task batches from `diskgraph-runtime` without
loading Python. It owns immutable task registration, epoch/shard iteration,
Arrow serialization, bounded production, cancellation, health counters, and
structured lifecycle/progress logs.

```bash
cargo run --release -p diskgraph-gateway -- \
  --data /local/graph --host 0.0.0.0 --port 8815 \
  --max-streams 1 --max-threads 2
```

The graph must already exist. The client registers its completed training-table
extension with `DoPut`; file names, lengths, order, SHA-256 digest, and final
completion are verified before an atomic rename publishes the task. Failed
uploads remain in their hidden staging directory for diagnosis.

Protocol version 5 uses a stable Arrow envelope containing JSON metadata and
Arrow IPC components. A `DoGet` ticket contains one normalized epoch/shard
request. The server reports sample/batch counts, then streams complete batches
and validates native shard completion. Errors propagate through final RPC status.

A one-slot native output queue reserves capacity before sampling each batch.
Batch envelopes retain the 64 MiB payload limit. Each client prefetches at most
one decoded batch per server. When the trainer pauses, client prefetch and gRPC
buffers fill, Flight stops draining the output queue, and the native producer
waits for capacity. There are no application credit messages. These queue limits
exclude additional transport buffers and native sampling/encoding working memory.

Cancelling a reader wakes a producer blocked on output capacity and releases its
stream permit. Native sampling already in progress stops at the next batch
boundary. Listener shutdown cancels all producers; a transport that is not being
read may consume the existing five-second graceful-drain period before the
listener force-closes it.

`health` and `task_status` are the supported actions. The earlier per-request
graph/feature ticket API is replaced by complete task epochs. This service is
intended for trusted peers on a restricted network.

The Python extension exposes `diskgraph.SamplingServer`, a lifecycle handle for
this same Rust implementation. It runs inside the Python/Ray actor process on
Rust-owned threads. Constructor and close release the GIL; sampled batches never
pass through a Python server handler. The core `SamplingService` wrapper owns
this handle and advertises the worker's routable IP.

Explicit close cancels producers and joins native serving threads. Context-manager
exit closes the handle; dropping it signals asynchronous shutdown without
blocking Python garbage collection. Native events use the extension's existing
Python logging bridge. The CLI uses stderr tracing and handles SIGTERM/Ctrl-C.

From `packages/diskgraph`, validate with:

```bash
cargo test -p diskgraph-gateway
cargo clippy -p diskgraph-gateway --all-targets -- -D warnings
```

Python integration tests exercise the embedded listener and compare complete
batches with the local iterator. The standalone executable remains available
through Cargo; Python workers use the extension directly.
