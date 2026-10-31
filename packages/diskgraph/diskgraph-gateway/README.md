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

Protocol version 4 uses a stable Arrow envelope containing JSON metadata and
Arrow IPC components. `DoExchange` accepts one normalized epoch request. The
server reports sample/batch counts and produces batches only with consumer
credits (at most two per credit frame). It validates stream completion and
propagates native errors through the final RPC status. The bounded output channel
and stream semaphore limit queued work. Cancellation wakes a producer awaiting
credits or output capacity; native sampling stops at batch boundaries.

`health` and `task_status` are the supported actions. The earlier per-request
graph/feature ticket API is replaced by complete task epochs. This service is
intended for trusted peers on a restricted network.

The wheel bundles this executable alongside `ingest`. Python's `SamplingService`
launches it and retains its stdout/stderr log and atomic readiness record beside
the graph. `--port 0 --ready-file PATH` publishes the assigned port and process id
for owners that launch it directly. SIGTERM and Ctrl-C stop waiting producers and
shut down the listener.
The Python launcher also supplies `--parent-pid` so an abrupt worker exit stops
its native listener.

From `packages/diskgraph`, validate with:

```bash
cargo test -p diskgraph-gateway
cargo clippy -p diskgraph-gateway --all-targets -- -D warnings
```

Python transport integration tests launch the binary and compare its complete
batches with the local iterator. For a development build, set
`DISKGRAPH_TASK_GATEWAY` to the built executable before running those suites.
