# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional replicated Ray Core sampling with trainer-owned seed scheduling.

Launchers own snapshot leases and Ray lifetime. Actors use local snapshot paths;
Trainers select explicit seeds and receive complete Arrow batches directly.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from kumo_core.diskgraph.batch_contract import (
    ArtifactIdentity,
    BatchPayload,
    BatchRequest,
    ConsumerToken,
    SamplingContractError,
    canonical_json,
    content_digest,
)

logger = logging.getLogger(__name__)
RAY_VERSION = "2.59.0"


class SamplingUnavailableError(RuntimeError):
    """No compatible healthy replica can satisfy the request within its budget."""


def _ray() -> Any:
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
    session: object
    artifacts: ArtifactIdentity
    engine_seed: int
    num_threads: int
    job: str
    generation: int = 0
    admission_timeout: float = 60.0
    request_timeout: float = 300.0
    max_retries: int = 1

    def __post_init__(self) -> None:
        if (
            not self.job
            or self.generation < 0
            or self.max_retries < 0
            or self.engine_seed < 0
            or self.num_threads < 1
            or any(
                not math.isfinite(value) or value <= 0
                for value in (self.admission_timeout, self.request_timeout)
            )
        ):
            raise SamplingContractError("invalid Ray backend identity or limits")
        if not isinstance(self.session, bytes):
            object.__setattr__(self, "session", _ray().cloudpickle.dumps(self.session))


@dataclass(frozen=True)
class ExecutionSlot:
    """A reserved feeder; only its confirmed execution can release active work."""

    lease_id: str
    replica_id: int
    actor: Any


@dataclass
class _Lease:
    request_id: str
    consumer: ConsumerToken
    replica_id: int
    expires: float
    active: bool = False
    watched: bool = False
    cancelling: bool = False


class _SamplingSession:
    """Job-wide admission metadata; batch payloads never pass through this actor."""

    def __init__(self, replicas: list[Any], config: ReplicaConfig) -> None:
        self.replicas = dict(enumerate(replicas))
        self.config = config
        self.available = list(self.replicas)
        self.leases: dict[str, _Lease] = {}
        self.generations: dict[tuple[str, int, str], int] = {}
        self.condition = asyncio.Condition()
        self.closed = False

    def _validate(self, request: BatchRequest) -> None:
        request.validate()
        if self.closed:
            raise SamplingUnavailableError("sampling session is closed")
        if (
            request.artifacts != self.config.artifacts
            or request.engine_seed != self.config.engine_seed
            or request.num_threads != self.config.num_threads
        ):
            raise SamplingContractError("request differs from sampler placement")
        token = request.consumer
        key = (token.job, token.trainer_rank, token.consumer_kind)
        if token.generation != self.generations.get(key, token.generation):
            raise SamplingContractError("obsolete consumer generation")
        self.generations.setdefault(key, token.generation)

    def _release(self, lease_id: str) -> None:
        lease = self.leases.pop(lease_id)
        if lease.replica_id in self.replicas:
            self.available.append(lease.replica_id)

    async def acquire(self, request: BatchRequest, timeout: float) -> ExecutionSlot:
        """Wait for shared capacity, holding only small request metadata."""
        deadline = time.monotonic() + timeout
        async with self.condition:
            while True:
                self._validate(request)
                now = time.monotonic()
                # Reserved calls have not started. Expiry fences their late start.
                for lease_id, lease in list(self.leases.items()):
                    if not lease.active and not lease.watched and lease.expires <= now:
                        self._release(lease_id)
                if not self.replicas:
                    raise SamplingUnavailableError("no healthy sampling replicas")
                if self.available:
                    replica_id = self.available.pop()
                    lease_id = uuid4().hex
                    self.leases[lease_id] = _Lease(
                        request.request_id,
                        request.consumer,
                        replica_id,
                        now + timeout,
                    )
                    return ExecutionSlot(
                        lease_id, replica_id, self.replicas[replica_id]
                    )
                remaining = deadline - now
                if remaining <= 0:
                    raise SamplingUnavailableError(
                        "sampling admission deadline exceeded"
                    )
                expiries = [
                    lease.expires - now
                    for lease in self.leases.values()
                    if not lease.active and not lease.watched
                ]
                wait = min([remaining, *expiries])
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=wait)
                except TimeoutError:
                    continue

    async def start(self, slot: ExecutionSlot, request: BatchRequest) -> None:
        """Fence expired reservations before a feeder touches native state."""
        async with self.condition:
            self._validate(request)
            lease = self.leases.get(slot.lease_id)
            if (
                lease is None
                or lease.active
                or lease.cancelling
                or lease.expires <= time.monotonic()
                or lease.request_id != request.request_id
                or lease.replica_id != slot.replica_id
            ):
                raise SamplingContractError("expired or invalid feeder reservation")
            lease.active = True

    async def watch(self, lease_id: str, refs: list[Any]) -> None:
        """Release capacity after publication without fetching the batch here."""
        lease = self.leases.get(lease_id)
        if lease is None:
            return
        lease.watched = True
        # Nesting the ObjectRef avoids Ray's top-level argument dereferencing.
        # wait(fetch_local=False) observes publication, without transferring data
        # to the session or materializing the payload in this process.
        ray = _ray()
        await asyncio.to_thread(ray.wait, refs, num_returns=1, fetch_local=False)
        replica = self.replicas.get(lease.replica_id)
        if lease.cancelling and lease.active and replica is not None:
            # A cancelled result can become ready while native code still runs.
            # This method executes after that call on the same serial actor.
            try:
                await asyncio.to_thread(ray.get, replica.drain.remote())
            except ray.exceptions.RayActorError:
                logger.warning(
                    "Cancelled sampler %s died while draining", lease.replica_id
                )
                await self.actor_failed(lease.replica_id)
                return
        await self.complete(lease_id)

    async def open_consumer(self, token: ConsumerToken) -> ConsumerToken:
        """Join the current generation when opening a trainer-owned iterator.

        Args:
            token: Configured consumer identity, possibly from an earlier iterator.

        Returns:
            Consumer identity with the session's current generation.

        Raises:
            SamplingUnavailableError: If the session is closed.
        """
        if self.closed:
            raise SamplingUnavailableError("sampling session is closed")
        key = (token.job, token.trainer_rank, token.consumer_kind)
        generation = self.generations.setdefault(key, token.generation)
        return replace(token, generation=generation)

    async def cancel_consumers(self, consumers: list[ConsumerToken]) -> None:
        """Fence iterators before their trainer cancels outstanding result refs.

        Active leases remain reserved until their completion watcher confirms
        that native execution has exited. Unstarted leases cannot pass start().
        Callers must await this acknowledgment before invoking ray.cancel(),
        then await their completion watchers before releasing snapshot users.

        Args:
            consumers: Exact consumer generations being closed by the trainer.
        """
        for token in consumers:
            await self.fence(token)
        cancelled = set(consumers)
        async with self.condition:
            for lease_id, lease in list(self.leases.items()):
                if lease.consumer in cancelled:
                    lease.cancelling = True
                    if not lease.active:
                        self._release(lease_id)
            self.condition.notify_all()

    async def complete(self, lease_id: str) -> None:
        """Release a slot only after native execution and export have ended."""
        async with self.condition:
            if lease_id in self.leases:
                self._release(lease_id)
                self.condition.notify_all()

    async def actor_failed(self, replica_id: int) -> None:
        """Remove a confirmed dead actor and its execution slots."""
        async with self.condition:
            self.replicas.pop(replica_id, None)
            self.available = [i for i in self.available if i != replica_id]
            for lease_id, lease in list(self.leases.items()):
                if lease.replica_id == replica_id:
                    self.leases.pop(lease_id)
            self.condition.notify_all()

    async def validate_delivery(self, request: BatchRequest) -> None:
        """Reject late delivery after cancellation, without fencing other consumers."""
        self._validate(request)

    async def fence(self, token: ConsumerToken) -> None:
        """Stop admission/delivery for this generation; active slots stay occupied."""
        async with self.condition:
            key = (token.job, token.trainer_rank, token.consumer_kind)
            self.generations[key] = max(
                self.generations.get(key, token.generation),
                token.generation + 1,
            )
            self.condition.notify_all()

    async def stats(self) -> dict[str, int]:
        """Return admission counters without inspecting or fetching batch data."""
        return {
            "replicas": len(self.replicas),
            "available": len(self.available),
            "reserved": sum(not lease.active for lease in self.leases.values()),
            "active": sum(lease.active for lease in self.leases.values()),
        }

    async def close(self) -> None:
        """Stop all admission/delivery; the launcher must drain or terminate actors."""
        async with self.condition:
            self.closed = True
            self.condition.notify_all()


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
        self.contexts: OrderedDict[bytes, Any] = OrderedDict()

    def ready(self) -> ReplicaConfig:
        """Return the verified local placement after native engines are opened."""
        return self.config

    def drain(self) -> None:
        """Acknowledge that earlier sampling on this serial actor has exited."""

    def sample(
        self, request: BatchRequest, slot: ExecutionSlot, session: Any
    ) -> BatchPayload:
        """Return a complete Arrow/native envelope directly to the requesting trainer."""
        ray = _ray()
        ray.get(session.start.remote(slot, request))
        if (
            request.artifacts != self.config.artifacts
            or request.engine_seed != self.config.engine_seed
            or request.num_threads != self.config.num_threads
        ):
            raise SamplingContractError("request does not match replica artifacts/RNG")
        options = request.validate()
        context = self.contexts.get(request.options_json)
        if context is None:
            context, _ = self.index.iter(
                **options,
                num_workers=1,
                worker_id=0,
                shuffle=False,
                epoch=0,
            )
            self.contexts[request.options_json] = context
            if len(self.contexts) > self.config.context_capacity:
                self.contexts.popitem(last=False)
        else:
            self.contexts.move_to_end(request.options_json)
        started = time.monotonic()
        batch = context.sample_request(list(request.seed_ids))
        payload = BatchPayload(request, batch, time.monotonic() - started)
        payload.validate(request)
        return payload


def start_sampling_session(
    configs: list[ReplicaConfig], placements: list[object]
) -> tuple[Any, tuple[Any, ...]]:
    """Create snapshot-local serial actors and a shared admission session.

    Args:
        configs: Verified-manifest descriptors for complete local replicas.
        placements: Explicit Ray scheduling strategies, one per local placement.

    Returns:
        Session handle and replica handles owned by the caller.

    Raises:
        SamplingContractError: If replica identities/resources disagree.
        RuntimeError: If any actor fails readiness; created actors are terminated.
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
    ray = _ray()
    actors = []
    with ExitStack() as cleanup:
        actor_class = ray.remote(_SamplerReplica)
        for config, placement in zip(configs, placements, strict=True):
            actor = actor_class.options(
                num_cpus=config.num_threads,
                max_concurrency=1,
                max_restarts=0,
                max_task_retries=0,
                scheduling_strategy=placement,
            ).remote(config)
            actors.append(actor)
            cleanup.callback(ray.kill, actor, no_restart=True)
        ray.get([actor.ready.remote() for actor in actors], timeout=300)
        session = (
            ray.remote(_SamplingSession)
            .options(
                num_cpus=0,
                max_restarts=0,
                max_task_retries=0,
            )
            .remote(actors, baseline)
        )
        cleanup.pop_all()
        return session, tuple(actors)


def stop_sampling_session(session: Any, replicas: tuple[Any, ...]) -> None:
    """Fence the whole job and terminate its actors before releasing snapshot leases.

    Args:
        session: Admission session returned by ``start_sampling_session``.
        replicas: All sampler handles belonging to the job.

    Raises:
        RuntimeError: If the session cannot acknowledge shutdown.
    """
    ray = _ray()
    try:
        ray.get(session.close.remote(), timeout=30)
    finally:
        for actor in replicas:
            ray.kill(actor, no_restart=True)
        ray.kill(session, no_restart=True)


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
        config.connection.connect()
        self.session = _ray().cloudpickle.loads(config.session)

    def iter(self, **options: Any) -> tuple[Iterator[Mapping[str, Any]], int]:
        """Create a seed iterator without opening local adjacency/features.

        Args:
            **options: Existing native iterator options and logical shard identity.

        Returns:
            Synchronous remote-batch iterator and worker-local sample/bucket count.

        Raises:
            SamplingContractError: If inherited through fork or options are invalid.
        """
        requests, count = self.iter_requests(**options)
        return map(self._sample, requests), count

    def iter_requests(self, **options: Any) -> tuple[Iterator[BatchRequest], int]:
        """Schedule logical requests without submitting remote sampling work.

        Args:
            **options: Native iterator options and logical shard identity.

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
        options = {
            **options,
            "rnd_seed": options.get("rnd_seed", self.config.engine_seed),
        }
        schedule, count = make_seed_schedule(self.manifest_path, options)
        # Selection is trainer-owned; feeders receive only fixed sampling options.
        scheduling = {
            "num_workers",
            "worker_id",
            "shuffle",
            "epoch",
            "drop_last",
            "group_by_timestamp",
            "weighted_seed_sampling",
            "selected_stage_node_ids",
        }
        native_options = {
            key: value for key, value in options.items() if key not in scheduling
        }
        return self._requests(schedule, options, canonical_json(native_options)), count

    def _requests(
        self, schedule: Any, options: Mapping[str, Any], native_options: bytes
    ) -> Iterator[BatchRequest]:
        ordinal = 0
        while (seeds := schedule.next_seed_ids()) is not None:
            yield BatchRequest(
                self.config.artifacts,
                self.consumer,
                options["epoch"],
                options["worker_id"],
                options["num_workers"],
                ordinal,
                tuple(seeds),
                native_options,
                self.config.engine_seed,
                self.config.num_threads,
            )
            ordinal += 1

    def _sample(self, request: BatchRequest) -> Mapping[str, Any]:
        ray = _ray()
        request.validate()
        for attempt in range(self.config.max_retries + 1):
            slot = ray.get(
                self.session.acquire.remote(
                    request,
                    self.config.admission_timeout,
                )
            )
            ref = slot.actor.sample.remote(request, slot, self.session)
            completion = self.session.watch.remote(slot.lease_id, [ref])
            try:
                payload = ray.get(ref, timeout=self.config.request_timeout)
            except ray.exceptions.ActorDiedError:
                # Restarts and automatic method retries are disabled. A terminal
                # ActorDiedError confirms the previous attempt can no longer run.
                ray.get(self.session.actor_failed.remote(slot.replica_id))
                logger.error(
                    "Sampling actor died during request %s", request.request_id
                )
                if attempt == self.config.max_retries:
                    raise
                continue
            except ray.exceptions.ObjectLostError:
                # Observe completion without fetching the payload at the session.
                # The failed result has not been delivered to the optimizer.
                ray.get(completion)
                logger.error("Sampling result lost for request %s", request.request_id)
                if attempt == self.config.max_retries:
                    raise
                continue
            except ray.exceptions.GetTimeoutError:
                # Slow is not dead: fence delivery and leave the slot owned by
                # the executing actor. Recovery requires the job checkpoint.
                ray.get(self.session.fence.remote(request.consumer))
                raise
            ray.get(self.session.validate_delivery.remote(request))
            return payload.validate(request)
        raise SamplingUnavailableError("sampling attempts exhausted")
