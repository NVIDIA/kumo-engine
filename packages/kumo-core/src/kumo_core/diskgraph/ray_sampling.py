# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional replicated Ray Core sampling with trainer-owned seed scheduling.

Launchers own snapshot leases and Ray lifetime. Actors use local snapshot paths;
Trainers select explicit seeds and receive encoded heterogeneous graphs directly.
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
from hashlib import sha256
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, cast

import pyarrow as pa
from diskgraph._c import SeedSchedule
from diskgraph.runtime_batch import RuntimeBatch
from diskgraph.sampling_options import InheritEngineSeed, SamplingOptions
from torch_geometric.data import HeteroData

from kumo_core.diskgraph.batch_contract import (
    ArtifactIdentity,
    BatchRequest,
    HeteroBatchPayload,
    SamplingContractError,
)
from kumo_core.diskgraph.hetero_wire import pack_python_hetero, unpack_python_hetero

if TYPE_CHECKING:
    from ray import ObjectRef
    from ray.util.scheduling_strategies import SchedulingStrategyT

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
    def sample(self) -> RemoteMethod[[BatchRequest], HeteroBatchPayload]:
        """Sample an explicit batch request."""
        ...

    @property
    def drain(self) -> RemoteMethod[[], None]:
        """Confirm completion of preceding serial native work."""
        ...

    @property
    def ready(self) -> RemoteMethod[[], None]:
        """Acknowledge verified replica initialization."""
        ...


class BatchContext(Protocol):
    """Native stateless sampling interface cached by each replica."""

    def sample_request(self, node_ids: list[int]) -> RuntimeBatch:
        """Sample the supplied canonical seed IDs without advancing a cursor."""
        ...


logger = logging.getLogger(__name__)


class SamplingUnavailableError(RuntimeError):
    """No compatible healthy replica can satisfy the request within its budget."""


def _ray() -> ModuleType:
    """Import optional Ray without changing its version or runtime lifecycle."""
    try:
        import ray
    except ModuleNotFoundError as error:
        if error.name != "ray":
            raise
        raise ModuleNotFoundError(
            "Remote sampling requires Ray; install kumo-core[ray].", name="ray"
        ) from error
    return ray


def _initialized_ray() -> ModuleType:
    """Require the caller to initialize Ray before using remote sampling."""
    ray = _ray()
    if not ray.is_initialized():
        raise RuntimeError("Initialize Ray before using remote sampling")
    return ray


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
    batch_encoder: Callable[[RuntimeBatch], HeteroData]
    context_capacity: int = 16

    def verify(self) -> None:
        """Validate manifest identities and CPU/context budgets before readiness.

        Raises:
            SamplingContractError: If artifacts or configuration differ.
            OSError: If local snapshot/task manifests cannot be read.
        """
        if min(self.num_threads, self.context_capacity) < 1 or self.engine_seed < 0:
            raise SamplingContractError("invalid replica thread/context/seed settings")
        if not callable(self.batch_encoder):
            raise SamplingContractError("sampler requires a batch encoder")
        for path, expected in (
            (self.snapshot_manifest_path, self.artifacts.snapshot_digest),
            (self.manifest_path, self.artifacts.task_digest),
        ):
            if sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise SamplingContractError(
                    "local snapshot/task manifest digest differs"
                )


@dataclass(frozen=True)
class RayBackendConfig:
    """Sampler handles and settings for a caller-initialized Ray runtime."""

    replicas: tuple[SamplerActor, ...]
    artifacts: ArtifactIdentity
    engine_seed: int
    num_threads: int
    request_timeout: float = 300.0
    max_retries: int = 1

    def __post_init__(self) -> None:
        if (
            self.max_retries < 0
            or self.engine_seed < 0
            or self.num_threads < 1
            or not math.isfinite(self.request_timeout)
            or self.request_timeout <= 0
        ):
            raise SamplingContractError("invalid Ray backend identity or limits")
        if not isinstance(self.replicas, tuple) or not self.replicas:
            raise SamplingContractError("sampling replicas must be a nonempty tuple")


class _SamplerPool:
    """Trainer-local actor availability for an exclusively owned replica set.

    Callers consume a result before refilling its actor. No admission or delivery
    RPC runs on a remote coordinator. Closing fences local delivery first, then
    drains serial actors because Ray cancellation alone does not prove native exit.
    """

    def __init__(self, config: RayBackendConfig) -> None:
        self.ray = _initialized_ray()
        self.config = config
        self.replicas = config.replicas
        self.available = deque(range(len(self.replicas)))
        self.pending: dict[ObjectRef[HeteroBatchPayload], int] = {}
        self.lock = Lock()
        self.close_lock = Lock()
        self.closed = False

    def submit(self, request: BatchRequest) -> ObjectRef[HeteroBatchPayload]:
        """Submit directly to an idle actor; callers bound their request window.

        Args:
            request: Explicit logical sampling request.

        Returns:
            Ray result reference for the submitted request.

        Raises:
            SamplingContractError: If the request uses incompatible RNG settings.
            SamplingUnavailableError: If closed or all actors have outstanding work.
        """
        request.validate()
        if (
            request.engine_seed != self.config.engine_seed
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

    def get(self, result: ObjectRef[HeteroBatchPayload]) -> HeteroBatchPayload:
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
                HeteroBatchPayload,
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
        self, result: ObjectRef[HeteroBatchPayload], failed: bool = False
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

    def ready(self) -> None:
        """Acknowledge initialization without returning fitted encoder state."""

    def drain(self) -> None:
        """Acknowledge that earlier sampling on this serial actor has exited."""

    def sample(self, request: BatchRequest) -> HeteroBatchPayload:
        """Sample and encode a graph before returning metadata and NumPy buffers."""
        if (
            request.engine_seed != self.config.engine_seed
            or request.num_threads != self.config.num_threads
        ):
            raise SamplingContractError("request does not match replica RNG settings")
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

        batch = context.sample_request(list(request.seed_ids))
        seed_ids = (
            batch.task.seed_node_ids
            if options.rhs_seed_node_type
            else batch.task.tt_stage_node_ids
        )
        if not seed_ids.equals(pa.array(request.seed_ids, type=pa.uint64())):
            raise SamplingContractError("sample seed membership/order differs")
        metadata, buffers = pack_python_hetero(self.config.batch_encoder(batch))
        return metadata, buffers


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
        RuntimeError: If Ray is uninitialized or actor readiness fails.
            Created actors are terminated after a readiness failure.
    """
    if not configs or len(configs) != len(placements):
        raise SamplingContractError("one explicit placement is required per replica")
    baseline = configs[0]
    if any(
        (c.artifacts, c.engine_seed, c.num_threads)
        != (
            baseline.artifacts,
            baseline.engine_seed,
            baseline.num_threads,
        )
        for c in configs
    ):
        raise SamplingContractError(
            "replicas must have identical artifacts and RNG pools"
        )
    ray = _initialized_ray()
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

    Raises:
        RuntimeError: If Ray is not initialized and replicas need to be stopped.
    """
    if not replicas:
        return
    ray = _initialized_ray()
    for actor in replicas:
        ray.kill(actor, no_restart=True)


class RayRuntimeIndex:
    """Process-local client sharing the native seed schedule with local iteration."""

    def __init__(self, manifest_path: str, config: RayBackendConfig) -> None:
        self.manifest_path = manifest_path
        self.config = config
        self.pid = os.getpid()
        self._manifest_verified = False
        self.samplers = _SamplerPool(config)

    def iter(self, options: SamplingOptions) -> tuple[Iterator[HeteroData], int]:
        """Receive actor-encoded graphs from typed metadata and NumPy buffers.

        Args:
            options: DiskGraph sampling configuration and logical shard identity.

        Returns:
            Decoded graph iterator and worker-local sample/bucket count.

        Raises:
            SamplingContractError: If the response is not an encoded graph.
        """
        requests, count = self.iter_requests(options)
        return map(self._sample, requests), count

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
            digest = sha256(Path(self.manifest_path).read_bytes()).hexdigest()
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

    def _sample(self, request: BatchRequest) -> HeteroData:
        payload = self._receive(request)
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise SamplingContractError("sampler did not return an encoded graph")
        return unpack_python_hetero(*payload)

    def _receive(self, request: BatchRequest) -> HeteroBatchPayload:
        ray = _ray()
        for attempt in range(self.config.max_retries + 1):
            ref = self.samplers.submit(request)
            try:
                payload = self.samplers.get(ref)
            except (ray.exceptions.ActorDiedError, ray.exceptions.ObjectLostError):
                logger.error(
                    "Sampling attempt failed: epoch=%d shard=%d batch=%d",
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
