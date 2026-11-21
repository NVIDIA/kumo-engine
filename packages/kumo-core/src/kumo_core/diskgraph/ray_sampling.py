# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional replicated Ray Core sampling with trainer-owned seed scheduling.

Launchers own snapshot leases and Ray lifetime. Actors use local snapshot paths;
Trainers select explicit seeds and receive complete Arrow batches directly.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, cast

from diskgraph.runtime_batch import RuntimeBatch
from diskgraph.sampling_options import InheritEngineSeed, SamplingOptions

from kumo_core.diskgraph.batch_contract import (
    ArtifactIdentity,
    BatchPayload,
    BatchRequest,
    ConsumerToken,
    HeteroBatchPayload,
    SamplingContractError,
    content_digest,
)

if TYPE_CHECKING:
    from diskgraph._c import SeedSchedule
    from ray import ObjectRef
    from ray.util.scheduling_strategies import SchedulingStrategyT
    from torch_geometric.data import HeteroData

_MethodArgs = ParamSpec("_MethodArgs")
_MethodResult = TypeVar("_MethodResult")


class RemoteMethod(Protocol[_MethodArgs, _MethodResult]):
    """Typed invocation surface for a bound Ray actor method."""

    def remote(
        self, *args: _MethodArgs.args, **kwargs: _MethodArgs.kwargs
    ) -> ObjectRef[_MethodResult]:
        """Submit a method call and return its typed Ray reference."""
        ...


class SamplerActor(Protocol):
    """Only the remote operations required from an owned sampler actor."""

    @property
    def sample(self) -> RemoteMethod[[BatchRequest], BatchPayload | HeteroBatchPayload]:
        """Sample an explicit batch request."""
        ...

    @property
    def drain(self) -> RemoteMethod[[], None]:
        """Confirm completion of preceding serial native work."""
        ...

    @property
    def ready(self) -> RemoteMethod[[], ReplicaConfig]:
        """Read the verified replica configuration."""
        ...


class BatchContext(Protocol):
    """Native stateless sampling interface cached by each replica."""

    def sample_request(self, node_ids: list[int]) -> RuntimeBatch:
        """Sample the supplied canonical seed IDs without advancing a cursor."""
        ...


logger = logging.getLogger(__name__)
RAY_VERSION = "2.59.0"


class SamplingUnavailableError(RuntimeError):
    """No compatible healthy replica can satisfy the request within its budget."""


def _ray() -> ModuleType:
    import ray

    if ray.__version__ != RAY_VERSION:
        raise SamplingContractError(f"Ray {RAY_VERSION} is required on every process")
    return ray


@dataclass(frozen=True)
class RayConnection:
    """Serializable cluster connection for a process-local Ray client."""

    address: str
    namespace: str

    def connect(self) -> None:
        """Attach this process to the configured cluster if needed.

        Raises:
            SamplingContractError: If the Ray version or namespace differs.
            ConnectionError: If the cluster is unavailable.
        """
        ray = _ray()
        if not ray.is_initialized():
            ray.init(address=self.address, namespace=self.namespace)
        if ray.get_runtime_context().namespace != self.namespace:
            raise SamplingContractError("client attached to a different Ray namespace")


@dataclass(frozen=True)
class ReplicaConfig:
    """Local placement and immutable artifacts for one persistent CPU sampler.

    The runtime manifest must have identical bytes across placements. Paths in
    that manifest are mounted consistently. Snapshot-manifest bytes must bind
    the graph's immutable file checksums; the launcher retains its snapshot lease.
    """

    manifest_path: str
    snapshot_manifest_path: str
    artifacts: ArtifactIdentity
    engine_seed: int
    num_threads: int
    context_capacity: int = 16
    batch_encoder: Callable[[RuntimeBatch], HeteroData] | None = None

    def verify(self) -> None:
        """Validate manifest identities and CPU/context budgets before readiness.

        Raises:
            SamplingContractError: If artifacts or configuration differ.
            OSError: If local snapshot/task manifests cannot be read.
        """
        if min(self.num_threads, self.context_capacity) < 1 or self.engine_seed < 0:
            raise SamplingContractError("invalid replica thread/context/seed settings")
        for path, expected in (
            (self.snapshot_manifest_path, self.artifacts.snapshot_digest),
            (self.manifest_path, self.artifacts.task_digest),
        ):
            if content_digest(Path(path).read_bytes()) != expected:
                raise SamplingContractError(
                    "local snapshot/task manifest digest differs"
                )


@dataclass(frozen=True)
class RayBackendConfig:
    """Serializable backend descriptor; actor handles attach after Ray connects."""

    connection: RayConnection
    # Delay deserializing actor handles until the receiving process connects.
    replicas: bytes | tuple[SamplerActor, ...]
    artifacts: ArtifactIdentity
    engine_seed: int
    num_threads: int
    job: str
    generation: int = 0
    request_timeout: float = 300.0
    max_retries: int = 1

    def __post_init__(self) -> None:
        if (
            not self.job
            or self.generation < 0
            or self.max_retries < 0
            or self.engine_seed < 0
            or self.num_threads < 1
            or not math.isfinite(self.request_timeout)
            or self.request_timeout <= 0
        ):
            raise SamplingContractError("invalid Ray backend identity or limits")
        if not isinstance(self.replicas, bytes):
            if not isinstance(self.replicas, tuple) or not self.replicas:
                raise SamplingContractError(
                    "sampling replicas must be a nonempty tuple"
                )
            object.__setattr__(
                self, "replicas", _ray().cloudpickle.dumps(self.replicas)
            )


class _SamplerPool:
    """Trainer-local actor availability for an exclusively owned replica set.

    Callers consume a result before refilling its actor. No admission or delivery
    RPC runs on a remote coordinator. Closing fences local delivery first, then
    drains serial actors because Ray cancellation alone does not prove native exit.
    """

    def __init__(self, config: RayBackendConfig) -> None:
        config.connection.connect()
        self.ray = _ray()
        self.config = config
        encoded = config.replicas
        self.replicas = (
            cast(tuple[SamplerActor, ...], self.ray.cloudpickle.loads(encoded))
            if isinstance(encoded, bytes)
            else encoded
        )
        if not self.replicas:
            raise SamplingContractError("at least one sampling replica is required")
        self.available = deque(range(len(self.replicas)))
        self.pending: dict[ObjectRef[BatchPayload | HeteroBatchPayload], int] = {}
        self.lock = Lock()
        self.close_lock = Lock()
        self.closed = False

    def submit(
        self, request: BatchRequest
    ) -> ObjectRef[BatchPayload | HeteroBatchPayload]:
        """Submit directly to an idle actor; callers bound their request window.

        Args:
            request: Explicit logical sampling request.

        Returns:
            Ray result reference for the submitted request.

        Raises:
            SamplingContractError: If the request uses incompatible artifacts/RNG.
            SamplingUnavailableError: If closed or all actors have outstanding work.
        """
        request.validate()
        if (
            request.artifacts != self.config.artifacts
            or request.engine_seed != self.config.engine_seed
            or request.num_threads != self.config.num_threads
        ):
            raise SamplingContractError("request differs from sampler configuration")
        with self.lock:
            if self.closed:
                raise SamplingUnavailableError("sampling dispatcher is closed")
            if not self.available:
                raise SamplingUnavailableError(
                    "consume a result before refilling samplers"
                )
            replica_id = self.available[0]
            result = self.replicas[replica_id].sample.remote(request)
            self.available.popleft()
            self.pending[result] = replica_id
            return result

    def get(
        self, result: ObjectRef[BatchPayload | HeteroBatchPayload]
    ) -> BatchPayload | HeteroBatchPayload:
        """Receive a result and make only its completed actor available again.

        Args:
            result: Reference previously returned by ``submit``.

        Returns:
            Unmodified sampler payload.

        Raises:
            SamplingUnavailableError: If this dispatcher has been cancelled.
            ray.exceptions.RayError: On delivery failure; timeouts retain ownership.
        """
        with self.lock:
            if self.closed:
                raise SamplingUnavailableError("sampling dispatcher is closed")
            if result not in self.pending:
                raise SamplingContractError("unknown sampling result")
        try:
            payload = cast(
                BatchPayload | HeteroBatchPayload,
                self.ray.get(result, timeout=self.config.request_timeout),
            )
        except self.ray.exceptions.ActorDiedError:
            self._finished(result, failed=True)
            raise
        except (self.ray.exceptions.ObjectLostError, self.ray.exceptions.RayTaskError):
            self._finished(result)
            raise
        # A timeout or other uncertain transport failure keeps the actor occupied.
        self._finished(result)
        with self.lock:
            if self.closed:
                raise SamplingUnavailableError("discarding cancelled sampling result")
        return payload

    def _finished(
        self, result: ObjectRef[BatchPayload | HeteroBatchPayload], failed: bool = False
    ) -> None:
        with self.lock:
            if not self.closed:
                replica_id = self.pending.pop(result)
                if not failed:
                    self.available.append(replica_id)

    def cancel(self) -> None:
        """Fence local submissions/delivery before cancelling outstanding Ray calls.

        Raises:
            ray.exceptions.RayError: If cancellation cannot be issued.
        """
        with self.lock:
            self.closed = True
            pending = tuple(self.pending)
        for result in pending:
            self.ray.cancel(result, force=False, recursive=False)

    def close(self) -> None:
        """Cancel and confirm native exit before another iterator reuses the actors.

        Raises:
            ray.exceptions.RayError: If live actors cannot drain before the deadline.
        """
        with self.close_lock:
            self.cancel()
            with self.lock:
                actors = {replica_id for replica_id in self.pending.values()}
            # Send every barrier before waiting; each runs after prior native work.
            barriers = [self.replicas[index].drain.remote() for index in actors]
            deadline = time.monotonic() + self.config.request_timeout
            for barrier in barriers:
                try:
                    self.ray.get(barrier, timeout=max(0.0, deadline - time.monotonic()))
                except self.ray.exceptions.ActorDiedError:
                    logger.warning("Sampling actor died while draining cancelled work")
            with self.lock:
                self.pending.clear()
                self.available.clear()


class _SamplerReplica:
    """One serial native engine per task extension, with a bounded context cache."""

    def __init__(self, config: ReplicaConfig) -> None:
        from diskgraph import RuntimeIndex

        config.verify()
        self.config = config
        self.index = RuntimeIndex(
            config.manifest_path,
            rnd_seed=config.engine_seed,
            num_threads=config.num_threads,
        )
        # Preopen every required task extension before reporting readiness.
        self.index._resolve()
        from diskgraph.runtime_batch_iterator import _extension_ref_from_stage_path

        for ref in self.index._tables.values():
            self.index._index_for(
                _extension_ref_from_stage_path(
                    self.index._parent_root,
                    ref["index_path"],
                )
            )
        self.contexts: list[tuple[SamplingOptions, BatchContext]] = []

    def ready(self) -> ReplicaConfig:
        """Return the verified local placement after native engines are opened."""
        return self.config

    def drain(self) -> None:
        """Acknowledge that earlier sampling on this serial actor has exited."""

    def sample(self, request: BatchRequest) -> BatchPayload | HeteroBatchPayload:
        """Return a complete Arrow/native envelope directly to the requesting trainer."""
        if (
            request.artifacts != self.config.artifacts
            or request.engine_seed != self.config.engine_seed
            or request.num_threads != self.config.num_threads
        ):
            raise SamplingContractError("request does not match replica artifacts/RNG")
        request.validate()
        options = request.options
        for index, (cached_options, context) in enumerate(self.contexts):
            if cached_options == options:
                self.contexts.append(self.contexts.pop(index))
                break
        else:
            native_context, _ = self.index.iter(options)
            context = cast(BatchContext, native_context)
            self.contexts.append((options, context))
            if len(self.contexts) > self.config.context_capacity:
                self.contexts.pop(0)
        started = time.monotonic()
        batch = context.sample_request(list(request.seed_ids))
        if self.config.batch_encoder is not None:
            from kumo_core.diskgraph.hetero_wire import pack_python_hetero

            metadata, buffers = pack_python_hetero(self.config.batch_encoder(batch))
            return HeteroBatchPayload(
                request, metadata, buffers, time.monotonic() - started
            )
        payload = BatchPayload(request, batch, time.monotonic() - started)
        payload.validate(request)
        return payload


def start_sampling_replicas(
    configs: list[ReplicaConfig], placements: list[SchedulingStrategyT]
) -> tuple[SamplerActor, ...]:
    """Create snapshot-local serial actors without a remote admission coordinator.

    Args:
        configs: Verified-manifest descriptors for complete local replicas.
        placements: Explicit Ray scheduling strategies, one per local placement.

    Returns:
        Replica handles exclusively owned by one trainer dispatcher.

    Raises:
        SamplingContractError: If replica identities/resources disagree.
        RuntimeError: If any actor fails readiness; created actors are terminated.
    """
    if not configs or len(configs) != len(placements):
        raise SamplingContractError("one explicit placement is required per replica")
    baseline = configs[0]
    if any(
        (c.artifacts, c.engine_seed, c.num_threads, c.batch_encoder is None)
        != (
            baseline.artifacts,
            baseline.engine_seed,
            baseline.num_threads,
            baseline.batch_encoder is None,
        )
        for c in configs
    ):
        raise SamplingContractError(
            "replicas must have identical artifacts and RNG pools"
        )
    ray = _ray()
    actors: list[SamplerActor] = []
    with ExitStack() as cleanup:
        actor_class = ray.remote(_SamplerReplica)
        for config, placement in zip(configs, placements, strict=True):
            actor = cast(
                SamplerActor,
                actor_class.options(
                    num_cpus=config.num_threads,
                    max_concurrency=1,
                    max_restarts=0,
                    max_task_retries=0,
                    scheduling_strategy=placement,
                ).remote(config),
            )
            actors.append(actor)
            cleanup.callback(ray.kill, actor, no_restart=True)
        ray.get([actor.ready.remote() for actor in actors], timeout=300)
        cleanup.pop_all()
        return tuple(actors)


def stop_sampling_replicas(replicas: tuple[SamplerActor, ...]) -> None:
    """Terminate owned actors before releasing their snapshot leases.

    Args:
        replicas: All sampler handles belonging to the job.
    """
    ray = _ray()
    for actor in replicas:
        ray.kill(actor, no_restart=True)


class RayRuntimeIndex:
    """Process-local client sharing the native seed schedule with local iteration."""

    def __init__(
        self, manifest_path: str, config: RayBackendConfig, consumer: ConsumerToken
    ) -> None:
        self.manifest_path = manifest_path
        self.config = config
        self.consumer = consumer
        self.pid = os.getpid()
        self._manifest_verified = False
        self.samplers = _SamplerPool(config)

    def iter(self, options: SamplingOptions) -> tuple[Iterator[RuntimeBatch], int]:
        """Create a seed iterator without opening local adjacency/features.

        Args:
            options: DiskGraph configuration and logical shard identity.

        Returns:
            Synchronous remote-batch iterator and worker-local sample/bucket count.

        Raises:
            SamplingContractError: If inherited through fork or options are invalid.
        """
        requests, count = self.iter_requests(options)
        return map(self._sample, requests), count

    def iter_hetero(self, options: SamplingOptions) -> tuple[Iterator[HeteroData], int]:
        """Receive actor-encoded graphs through MessagePack and NumPy buffers.

        Args:
            options: DiskGraph sampling configuration and logical shard identity.

        Returns:
            Decoded graph iterator and worker-local sample/bucket count.

        Raises:
            SamplingContractError: If actors do not have a batch encoder configured.
        """
        requests, count = self.iter_requests(options)
        return map(self._sample_hetero, requests), count

    def iter_requests(
        self, options: SamplingOptions
    ) -> tuple[Iterator[BatchRequest], int]:
        """Schedule logical requests without submitting remote sampling work.

        Args:
            options: DiskGraph configuration; omitted RNG seed inherits the engine seed.

        Returns:
            Lazy requests and the worker-local sample/bucket count.

        Raises:
            SamplingContractError: If inherited through fork or metadata differs.
        """
        from kumo_core.diskgraph.seed_schedule import make_seed_schedule

        if self.pid != os.getpid():
            raise SamplingContractError(
                "Ray clients cannot be inherited across processes"
            )
        if not self._manifest_verified:
            digest = content_digest(Path(self.manifest_path).read_bytes())
            if digest != self.config.artifacts.task_digest:
                raise SamplingContractError("trainer task manifest digest differs")
            self._manifest_verified = True
        configuration = (
            replace(options, rnd_seed=self.config.engine_seed)
            if options.rnd_seed is InheritEngineSeed.DEFAULT
            else options
        )
        schedule, count = make_seed_schedule(self.manifest_path, configuration)
        sampling = replace(
            configuration,
            num_workers=1,
            worker_id=0,
            shuffle=False,
            epoch=0,
            drop_last=False,
            group_by_timestamp=False,
            weighted_seed_sampling=None,
            selected_stage_node_ids=None,
        )
        return self._requests(schedule, configuration, sampling), count

    def _requests(
        self,
        schedule: SeedSchedule,
        options: SamplingOptions,
        native_options: SamplingOptions,
    ) -> Iterator[BatchRequest]:
        ordinal = 0
        while (seeds := schedule.next_seed_ids()) is not None:
            yield BatchRequest(
                self.config.artifacts,
                self.consumer,
                options.epoch,
                options.worker_id,
                options.num_workers,
                ordinal,
                tuple(seeds),
                native_options,
                self.config.engine_seed,
                self.config.num_threads,
            )
            ordinal += 1

    def _sample(self, request: BatchRequest) -> RuntimeBatch:
        payload = self._receive(request)
        if not isinstance(payload, BatchPayload):
            raise SamplingContractError(
                "encoded responses require iter_hetero(options)"
            )
        return payload.validate(request)

    def _sample_hetero(self, request: BatchRequest) -> HeteroData:
        payload = self._receive(request)
        if not isinstance(payload, HeteroBatchPayload):
            raise SamplingContractError("sampler did not return an encoded graph")
        return payload.decode(request)

    def _receive(self, request: BatchRequest) -> BatchPayload | HeteroBatchPayload:
        ray = _ray()
        for attempt in range(self.config.max_retries + 1):
            ref = self.samplers.submit(request)
            try:
                payload = self.samplers.get(ref)
            except (ray.exceptions.ActorDiedError, ray.exceptions.ObjectLostError):
                logger.error(
                    "Sampling attempt failed: consumer=%s epoch=%d shard=%d batch=%d",
                    request.consumer,
                    request.epoch,
                    request.logical_shard,
                    request.batch_ordinal,
                )
                if attempt == self.config.max_retries:
                    raise
                continue
            except ray.exceptions.GetTimeoutError:
                # Keep ownership until close() confirms actual native exit.
                self.samplers.cancel()
                raise
            return payload
        raise SamplingUnavailableError("sampling attempts exhausted")

    def close(self) -> None:
        """Fence delivery and drain outstanding sampling before replica reuse.

        Raises:
            ray.exceptions.RayError: If remote work cannot drain before the deadline.
        """
        self.samplers.close()
