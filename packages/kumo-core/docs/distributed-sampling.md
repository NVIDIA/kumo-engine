# Ray Core sampling backend

The optional `kumo-core[ray]` extra pins Ray 2.59.0. All participants must use
that version and a DiskGraph build with `SeedScheduleConfig`, `SeedSchedule`, and
`RuntimeBatchIterator.sample_request` support, plus DiskGraph-owned `SamplingPlan`,
`SamplingOptions`, and typed `RuntimeBatch` records from #454 (stacked on #449 and #445).

The default backend returns DiskGraph's complete typed `RuntimeBatch`, retaining
its Arrow buffers. Actors can also run a supplied Python batch encoder and return
MessagePack metadata plus NumPy buffers for direct Python HeteroData reconstruction.
The asynchronous trainer loader and training integration remain separate; this
backend does not change the existing DataLoader implementation.

## Contracts and execution

- `batch_contract.py` defines immutable artifact identities, consumer generations,
  explicit seed requests, and validated batch envelopes.
  Requests use DiskGraph's `SamplingOptions` and `WeightedSeedSampling` directly;
  kumo-core defines no duplicate configuration or keyword adapters. These frozen
  dataclasses contain ordinary lists/dictionaries, which callers must not mutate
  during iteration. The native `SamplingPlan` is immutable. Requests and cached
  contexts compare typed values; no plan JSON, copies, digests, or serialized
  cache keys are needed. The actor retains at most `context_capacity` contexts,
  using structural equality and least-recently-used eviction.
  Wire version 2 rejects the previous JSON-bytes request format. All participants
  must use the matching request implementation. DiskGraph owns the runtime batch,
  task, graph, edge, and feature classes; kumo-core defines no duplicate batch
  schemas. Fixed fields use attributes; named channels/namespaces remain dicts.
- `seed_schedule.py` constructs the native seed schedule from task metadata and
  seed sidecars. The trainer does not open adjacency or feature files.
- `ray_sampling.py` provides serial native sampler actors and trainer-local actor
  tracking. Typed raw or encoded graph payloads travel directly to the trainer, with no
  remote admission, start, completion-watcher, or delivery-validation RPCs.

The launcher owns artifact distribution, file-checksum verification, consistent
local paths, and the immutable snapshot lease. `ReplicaConfig.verify()` checks
snapshot/task manifest digests before opening a local graph. The manifest digest
alone does not verify all underlying graph files. The mapping digest identifies
caller-owned fitted mappings; this raw-batch backend does not load those mappings.

`start_sampling_replicas(configs, placements)` accepts one explicit Ray scheduling
strategy per replica. It reserves `num_threads` CPUs per serial actor, opens task
extensions before readiness, and keeps a bounded context cache (16 by default).
Replica artifacts, engine seed, and native thread counts must agree. Automatic
actor restart and method retry are disabled.

Create a `RayBackendConfig` from the replica handles, artifact identities, and a
`RayConnection`, then construct `RayRuntimeIndex` with a `ConsumerToken`.
`iter_requests(options)` lazily enumerates requests without sampling RPCs.
`iter(options)` is a synchronous reference client returning raw batches and
sample/bucket count, using native iterator stage, epoch, shard, selection, and
sampling options. Omitted RNG seeds inherit the backend engine seed; explicit
`None` keeps DiskGraph's existing unseeded semantics. Scheduling fields are reset
in actor requests without changing the caller's configuration. The client cannot
be inherited across processes. Remote
neighborhood/history decorators are not exposed by this interface.

## Optional encoded responses

Set `ReplicaConfig.batch_encoder` to a serializable callable taking a
`diskgraph.RuntimeBatch` and returning CPU `HeteroData`. It can reuse fitted Kumo
mappers through `runtime_batch_to_hetero`; configure the callable with
`attach_pkeys=False`. The launcher owns the fitted state and must keep it
consistent with `ArtifactIdentity.mapping_digest` across replicas.

The actor calls `pack_python_hetero` after encoding. `HeteroBatchPayload` carries
MessagePack graph metadata (including tensor shapes, dtypes, and buffer indices)
and separate contiguous NumPy arrays. Tensor contents are never packed into the
MessagePack bytes. Ray still serializes the outer envelope normally and transports
NumPy data through its out-of-band buffers.

Consume these actors with `RayRuntimeIndex.iter_hetero(options)`. Its synchronous
reference iterator validates request identity and reconstructs TensorFrames and
HeteroData in Python using `torch.from_numpy`; it does not spawn DataLoader
subprocesses or use Rust feature encoding/decoding. Use `iter(options)` with raw
actors; either entry point rejects the wrong response kind.

The version-1 graph wire supports CPU int64/float32/float64 tensors, numerical,
categorical, timestamp, and ragged multi-categorical TensorFrame features, node
sampling metadata, and edge indices/counts. Nonempty primary keys, global graph
attributes, extra edge attributes, and unsupported feature layouts fail explicitly.
Noncontiguous tensors require a contiguous copy; existing contiguous storage is
borrowed. Treat received buffers as read-only and clone before in-place writes.
Zero-copy on the receiver does not eliminate cross-host network transfer.

## Concurrency, cancellation, and ownership

Each replica set belongs exclusively to one trainer dispatcher. Its inflight
window is bounded by the actor count, and local tracking ensures at most one
outstanding call per actor. Consuming a result makes that actor available for the
next request. The synchronous reference client issues one call at a time; the
future async loader can submit to all actors and consume completion order.
Multiple independent trainers must use distinct replica sets.

Cancellation fences local submission/delivery first, then cancels outstanding
Ray references. A cancelled result can become ready before native execution
exits. `RayRuntimeIndex.close()` waits for a serial `drain()` barrier on each
outstanding actor before returning, so replacement iterators can safely reuse
those actors. A timeout retains actor ownership; retry close if draining times
out. Callers must close before restarting or releasing snapshot users.

Only confirmed actor death or object loss is eligible for the synchronous
client's bounded retry policy. There is no remote generation registry or token
lease. Requests retain consumer identities for response validation; the trainer
owns iterator cancellation and rejects late results locally.

`stop_sampling_replicas(replicas)` kills only the owned actors. The launcher
retains ownership of the Ray runtime and snapshot lease; release that lease
only after shutdown. This function does not call `ray.shutdown()`.

## Focused validation

With the matching native build and optional Ray extra installed, run from the
repository root:

```bash
PYTHONPATH=packages/kumo-core/src .venv/bin/python -m pytest -q \
  --confcutdir=packages/kumo-core/tests/distributed \
  packages/kumo-core/tests/distributed
PYTHONPATH=packages/kumo-core/src .venv/bin/python -m pytest -q \
  packages/diskgraph/diskgraph-pybind/python/tests/test_ray_sampling_backend.py
```

The core tests cover identities, metadata-only scheduling, actor availability,
local fencing, cancellation/drain races, retry limits, Arrow transport, and startup
cleanup. The native integration test compares remote/local typed batches for shuffled,
selected-seed, and RHS sampling, plus real actor encoding/NumPy delivery and Python
reconstruction. `test_hetero_transport.py` covers feature parity, buffer lifetime,
noncontiguous views, response modes, and malformed layouts; it requires the Torch
runtime. Tests start only local Ray clusters.
