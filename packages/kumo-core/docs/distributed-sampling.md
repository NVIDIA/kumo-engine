# Ray sampling and encoding backend

The optional `kumo-core[ray]` extra installs Ray without an exact version pin.
Ray 2.59.0 is the tested version; the backend does not enforce a version string.
The launcher owns compatible Ray installations across the cluster and must call
`ray.init(...)` in each trainer process before creating or consuming sampler
actors. Core sampling never initializes, reconnects, or shuts down Ray.
All participants need the DiskGraph-owned `SamplingPlan`, `SamplingOptions`, and `RuntimeBatch` APIs from
#454 (stacked on #449 and #445), plus the native stateless sampling operation.

Every remote actor samples and encodes a batch. The trainer receives frozen dataclass
graph metadata and separate NumPy tensor buffers, then reconstructs `HeteroData`.
There is no remote raw-batch mode or whole-HeteroData pickle mode. Local DiskGraph
iteration continues to return typed runtime batches for local collation.

## Actors and encoding

`ReplicaConfig.batch_encoder` is a required serializable callable taking a
DiskGraph `RuntimeBatch` and returning CPU `HeteroData`. The actor invokes it after
sampling, then calls `pack_python_hetero`. Existing fitted Kumo mappers can be
reused through `runtime_batch_to_hetero` with `attach_pkeys=False`; fitted state
must agree with `ArtifactIdentity.mapping_digest` across replicas. Encoder state
is sent at actor creation rather than in each request.

The launcher owns artifact distribution, file-checksum verification, consistent
local paths, and snapshot leases. `ReplicaConfig.verify()` checks snapshot/task
manifest digests and rejects a missing encoder before opening graph data. A
manifest digest alone does not verify every underlying graph file.

`start_sampling_replicas(configs, placements)` takes an explicit placement per
actor. Each serial actor reserves `num_threads` CPUs, preopens task extensions,
and caches at most `context_capacity` native iterators (16 by default). Contexts
use structural `SamplingOptions` equality with least-recently-used eviction.
Actor restart and automatic method retries are disabled.

## Requests and consumption

`batch_contract.py` defines artifact identities, explicit seed requests,
and the `HeteroBatchPayload` tuple type alias. DiskGraph owns sampling configuration and raw runtime
records; kumo-core defines no duplicate native batch schemas or JSON option
adapters. Artifact identities stay in replica/backend configuration and are not
repeated in batch requests. Local manifests are verified during initialization.
`BatchRequest.validate()` checks validity and returns `None`.

`seed_schedule.py` builds the native seed schedule using only task metadata and
seed sidecars. Remote training schedules never open graph adjacency/features. Omitted RNG
seeds inherit the backend engine seed; explicit `None` and zero preserve native
semantics. Sharding, epoch, shuffle, selection, and weighted scheduling stay on
the trainer and are reset in actor-side options. Options are shallow frozen:
callers must not mutate their projection/selection containers during iteration.

Create `RayBackendConfig` from the replica handles and artifact identities, then
construct `RayRuntimeIndex(manifest_path, config)`. The configuration stores actor
handles directly; it carries no address, namespace, or manually pickled handles.
The caller must initialize Ray before transferring actor handles to another
trainer process. Starting actors, consuming batches, and stopping owned actors
fail clearly when Ray is uninitialized instead of triggering implicit startup.
`iter(options)` returns a synchronous `HeteroData` iterator and sample/bucket
count. The trainer associates each response with its pending Ray request; the
response does not echo request metadata. It reconstructs tensors using
`torch.from_numpy` without running feature encoders. The former `iter_hetero`
entry point is unnecessary because every response is encoded.

`iter_requests(options)` enumerates logical requests without issuing sampling
RPCs. The synchronous reference iterator issues one call at a time. Neither
interface supports inheriting a Ray client across processes.

## DataModule and asynchronous delivery

`DiskgraphLightningDataModule.remote_batch_encoder()` captures its fitted Python
mappers, target encoder, schema, and timestamp settings in a `RuntimeBatchEncoder`.
Use it as `ReplicaConfig.batch_encoder`. It calls the same
`runtime_batch_to_hetero(..., attach_pkeys=False)` conversion as local training;
the actor performs sampling and column encoding serially in its own process.
It does not capture the DataModule, worker pools, or trainer connection.

Attach the resulting `RayBackendConfig` as the DataModule's `sampling_backend`
before constructing a training iterator, or supply it to the constructor. Engine
seed and DataModule base seed must match. Actor CPU/thread settings belong to
the backend and need not equal the trainer's local evaluation thread settings.

`DiskgraphTaskBatchIterable` is the common dataset for both paths: its iterator
returns local `RuntimeBatch` values or already encoded remote `HeteroData` values.
The shared DataModule collator encodes native batches and passes encoded graphs
through unchanged. The dataset owns its remote iterator and closes the previous
receiver before starting another epoch; no separate remote dataset wrapper is
needed.

The train loader becomes `RemoteDataLoader` from `lightning_data`, with
`num_workers=0` regardless of the local worker setting. This thin loader uses the
same dataset/collator and delegates remote shutdown to the dataset. One receiver
thread issues requests and uses
`ray.wait(..., num_returns=1)` to receive whichever actor finishes first. The
trainer main thread reconstructs `HeteroData` using NumPy-backed tensors; it does
not encode columns or send tensors through a multiprocessing queue. Ordinary
DataLoader pinning and Lightning's device transfer still run on the trainer.

The global window equals the number of sampler actors. A slot covers a pending
RPC, received buffers, and reconstruction until delivery to DataLoader. Releasing
it wakes the receiver immediately, including while the model processes the
previous batch. Local `num_workers` and `prefetch_factor` do not change this
remote window. Batches arrive in completion order, so graph contents and logical
seed membership match local sampling but optimizer order can differ.

The same iterable schedule handles stage mixing, rank sharding, shuffle epochs,
grouped timestamps, epoch caps/replay, weighted sampling, and feature projections.
Each rank needs an exclusively owned actor pool. Resetting the iterator closes
and drains the preceding epoch before reusing its actors. DataModule shutdown,
teardown, and existing Lightning lifecycle callbacks also close remote loaders,
including when Lightning stops before the dataset's natural end.

Validation, test, prediction, explanation, and auxiliary RHS loaders retain their
local paths and primary-key support. With a remote backend attached, these
DataModule loaders also use zero subprocesses to avoid forking an initialized Ray
client. Their graph artifacts must still be
available on the trainer. Remote explanation capture is rejected explicitly.
This integration changes the core DataModule, not the public SDK initialization
or cluster provisioning API.

After the launcher places immutable artifacts and calls `ray.init(...)`, wiring is:

```python
encoder = datamodule.remote_batch_encoder()
configs = [
    ReplicaConfig(
        manifest_path=task_path,
        snapshot_manifest_path=snapshot_path,
        artifacts=artifacts,
        engine_seed=datamodule.base_seed,
        num_threads=sampling_threads,
        batch_encoder=encoder,
    )
    for _ in placements
]
actors = start_sampling_replicas(configs, placements)
try:
    datamodule.sampling_backend = RayBackendConfig(
        replicas=actors,
        artifacts=artifacts,
        engine_seed=datamodule.base_seed,
        num_threads=sampling_threads,
    )
    trainer.fit(model, datamodule=datamodule)
finally:
    datamodule.shutdown_dataloader_workers()
    stop_sampling_replicas(actors)
```

Here `placements` selects the sampler nodes, each path identifies the matching
node-local artifact, and `artifacts.mapping_digest` identifies the fitted encoder
state. Launchers retain snapshot leases until loader drain and actor shutdown
succeed. Snapshot copying, artifact distribution, and Ray lifetime are separate.

## Wire format and ownership

The metadata records contain node/edge types, feature names, sampling counts, tensor
shapes/dtypes, and buffer indices. Tensor contents remain separate contiguous
NumPy arrays. Actors return `(metadata, buffers)` directly. Ray handles the
tuple, frozen metadata dataclasses, and out-of-band array serialization.
Metadata uses Ray's default Python-object serialization; buffers remain separate.
There is no application-level binary metadata codec or outer response wrapper.

Graph wire version 1 supports CPU int64/float32/float64 tensors and numerical,
categorical, timestamp, and ragged multi-categorical TensorFrame features.
Nonempty primary keys, global attributes, extra edge attributes, and unsupported
feature layouts fail explicitly. Noncontiguous tensors require a contiguous
copy; contiguous storage is borrowed. Received tensors retain NumPy owners.
Treat the received buffers as read-only and clone before in-place writes.
Receiver-side zero-copy does not eliminate cross-host network transfer.

Request wire version 2 and graph wire version 1 are independent. All participants
must run the matching implementation. Old raw or whole-graph responses are
rejected rather than routed to trainer-side feature encoding.

## Concurrency and cancellation

One trainer dispatcher exclusively owns each replica set and keeps at most one
outstanding RPC per actor. Consuming a result makes that actor available again.
Multiple independent trainers must use distinct replica sets.

Cancellation fences local submission/delivery, then cancels outstanding Ray refs.
A cancelled ref may become ready before native execution exits. `close()` waits
for serial actor drain barriers before permitting replica reuse. A drain timeout
retains ownership; retry close before replacing iterators or releasing snapshots.
Only confirmed actor death/object loss permits bounded retry of the same logical
request. There are no remote admission tokens or per-batch control RPCs.

`stop_sampling_replicas` kills only owned actors; it does not shut down the Ray
runtime. The launcher releases the snapshot lease after actor shutdown.

## Focused validation

With the matching native build, Ray extra, and CPU Torch runtime installed:

```bash
cd packages/kumo-core
PYTHONPATH=src python -m pytest -q --confcutdir=tests/distributed tests/distributed
```

From the repository root, run native actor parity:

```bash
PYTHONPATH=packages/kumo-core/src python -m pytest -q \
  packages/diskgraph/diskgraph-pybind/python/tests/test_ray_sampling_backend.py
PYTHONPATH=packages/kumo-core/src python -m pytest -q \
  packages/diskgraph/diskgraph-pybind/python/tests/test_ray_datamodule.py
```

Tests cover mandatory encoders, native seed membership before encoding, metadata
scheduling, context reuse, request tracking, feature/buffer parity, malformed
layouts, actor ownership, retry, cancellation/drain races, and real encoded Ray
delivery for shuffled, selected, and RHS samples. Async tests cover the aggregate
window, completion order, refill during model work, failure propagation, empty
epochs, cancellation, and retrying a failed drain. A native integration test
compares encoded batches and runs a two-epoch CPU Lightning fit with early limits.
Only local Ray clusters start. Artifact distribution, Rust feature encoding, and
native Rust decoding are excluded.
