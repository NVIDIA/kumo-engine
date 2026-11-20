# Ray Core sampling backend

The optional `kumo-core[ray]` extra pins Ray 2.59.0. All participants must use
that version and a DiskGraph build with `SeedSchedule` and
`RuntimeBatchIterator.sample_request` support.

This backend returns complete native Arrow batches. Feature encoding, hetero
assembly, asynchronous trainer prefetch, and training integration are separate
layers. It does not change the existing DataLoader implementation.

## Contracts and execution

- `batch_contract.py` defines immutable artifact identities, consumer generations,
  explicit seed requests, and validated batch envelopes.
  Request identity includes the ordered seeds and sampling configuration, so
  retries preserve the same logical batch regardless of actor placement.
- `seed_schedule.py` constructs the native seed schedule from task metadata and
  seed sidecars. The trainer does not open adjacency or feature files.
- `ray_sampling.py` provides serial native sampler actors and trainer-local actor
  tracking. Raw Arrow payloads travel directly from sampler to trainer, with no
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

Create a `RayBackendConfig` from the session, artifact identities, and a
`RayConnection`, then construct `RayRuntimeIndex` with a `ConsumerToken`.
`iter_requests(**options)` lazily enumerates requests without sampling RPCs.
`iter(**options)` is a synchronous reference client returning raw batches and
sample/bucket count, using native iterator stage, epoch, shard, selection, and
sampling options. The client cannot be inherited across processes. Remote
neighborhood/history decorators are not exposed by this interface.

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
cleanup. The native integration test compares remote and local Arrow batches for
shuffled, selected-seed, and RHS sampling. Tests start only local Ray clusters.
