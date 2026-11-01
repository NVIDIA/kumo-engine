# Native Flight gateway

`diskgraph-gateway` serves complete task batches from `diskgraph-runtime` without
loading Python. It owns epoch/shard iteration over prepared local artifacts,
Arrow serialization, bounded production, cancellation, health counters, and
structured lifecycle/progress logs.

```bash
cargo run --release -p diskgraph-gateway -- \
  --data /local/graph --host 0.0.0.0 --port 8815 \
  --max-streams 1 --max-threads 2
```

The graph must already exist. Before requesting an epoch, the caller must
materialize and verify its training-table extension beneath the graph at
`train_tables/<id>` or `pred_tables/<id>`. Artifact distribution and manifest
verification belong to the preparation layer. This gateway reads those files;
it does not upload or register them. Missing extension metadata returns
`FailedPrecondition` so the caller can finish materialization before retrying.

Protocol version 7 streams native Arrow component record batches. A `DoGet`
ticket contains one normalized epoch/shard request. The initial empty Arrow
schema carries JSON metadata under `kumo.sampling`: `version`, `num_samples`,
and `num_batches`. Each component then sends its own schema, optional dictionaries,
and record batch, allowing task arrays, targets, graph arrays and feature tables
to have different schemas and row counts. Flight `app_metadata` identifies the
logical batch, component position and component count; the first component's
record batch also carries the small JSON layout that assembles the native batch.
There are no binary wrapper columns, nested IPC streams, or iterator/end control
record batches. Flight EOF completes the stream, and the client validates its
received count and final RPC status. The server also verifies native counts.

`diskgraph.SamplingClient` uses Rust's low-level `FlightDataDecoder`, which
supports schema changes and resets dictionaries between schemas. The ordinary
PyArrow Flight reader assumes a fixed schema and cannot read this protocol.
The native client validates and assembles a complete minibatch before exposing
its arrays and record batches through the Arrow C Data Interface to Python.

A one-slot native output queue reserves capacity before sampling each batch.
Each logical minibatch has a cumulative 64 MiB wire-payload limit, including
component schemas, dictionaries and application metadata. Each client prefetches at most
one decoded batch per server. When the trainer pauses, client prefetch and gRPC
buffers fill, Flight stops draining the output queue, and the native producer
waits for capacity. There are no application credit messages. These queue limits
exclude additional transport buffers and native sampling/encoding working memory.

Cancelling a reader wakes a producer blocked on output capacity and releases its
stream permit. Native sampling already in progress stops at the next batch
boundary. Listener shutdown cancels all producers; a transport that is not being
read may consume the existing five-second graceful-drain period before the
listener force-closes it.
New requests allow up to one second for a producer slot to become available,
including release from a preceding cancelled stream, then report the server busy.

`health` is the supported action. The earlier per-request
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
