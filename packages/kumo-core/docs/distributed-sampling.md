# Ray Core sampling backend

The optional `kumo-core[ray]` extra pins Ray 2.59.0. All participants must use
that version and a DiskGraph build with `SeedSchedule` and
`RuntimeBatchIterator.sample_request` support.

This backend returns complete native Arrow batches. Feature encoding, hetero
assembly, asynchronous trainer prefetch, and training integration are separate
layers. It does not change the existing DataLoader implementation.

## Contracts and execution

- `batch_contract.py` defines immutable artifact identities, independently fenced
  consumer generations, explicit seed requests, and validated batch envelopes.
  Request identity includes the ordered seeds and sampling configuration, so
  retries preserve the same logical batch regardless of actor placement.
- `seed_schedule.py` constructs the native seed schedule from task metadata and
  seed sidecars. The trainer does not open adjacency or feature files.
- `ray_sampling.py` provides serial native sampler actors and a shared admission
  broker. The broker handles request metadata and completion references; raw
  Arrow payloads travel directly from sampler to trainer.

The launcher owns artifact distribution, file-checksum verification, consistent
local paths, and the immutable snapshot lease. `ReplicaConfig.verify()` checks
snapshot/task manifest digests before opening a local graph. The manifest digest
alone does not verify all underlying graph files. The mapping digest identifies
caller-owned fitted mappings; this raw-batch backend does not load those mappings.

`start_sampling_session(configs, placements)` accepts one explicit Ray scheduling
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

## Cancellation and ownership

Admission reserves one actor per active request. Reservations that never start
can expire; started requests retain their lease until completion or confirmed
actor death. A request timeout fences delivery and does not imply actor death.
Only confirmed actor death or object loss is eligible for the client's bounded
retry policy.

A caller managing concurrent requests must:

1. Await `session.cancel_consumers.remote(tokens)` to fence their generations.
2. Cancel the corresponding sampling references with Ray.
3. Allow the registered session watchers to finish. Cancellation can mark a Ray
   reference ready before native execution exits; a serial actor `drain()` call
   confirms exit before its lease becomes available again.

Register each submitted sampling reference using `session.watch(lease_id, [ref])`.
The list keeps the reference nested so the broker never resolves the payload.
Use `open_consumer` to obtain a new generation before restarting a cancelled
consumer. Other consumer kinds/ranks remain independent. Delivery validation
rejects stale results.

`stop_sampling_session(session, replicas)` fences the job and kills its owned
actors. The launcher retains ownership of the Ray runtime and snapshot lease;
release that lease only after shutdown. This function does not call
`ray.shutdown()`.

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

The core tests cover identities, metadata-only scheduling, admission, generation
fencing, cancellation/drain races, retry limits, Arrow transport, and startup
cleanup. The native integration test compares remote and local Arrow batches for
shuffled, selected-seed, and RHS sampling. Tests start only local Ray clusters.
