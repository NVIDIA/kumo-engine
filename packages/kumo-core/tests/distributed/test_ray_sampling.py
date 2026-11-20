# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampling admission, immutable envelopes, and real Ray Arrow serialization."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from kumo_core.diskgraph.batch_contract import (
    ArtifactIdentity,
    BatchPayload,
    BatchRequest,
    ConsumerToken,
    SamplingContractError,
    canonical_json,
    content_digest,
)
from kumo_core.diskgraph.ray_sampling import (
    RAY_VERSION,
    RayBackendConfig,
    RayConnection,
    RayRuntimeIndex,
    ReplicaConfig,
    SamplingUnavailableError,
    _SamplerReplica,
    _SamplingSession,
    start_sampling_session,
    stop_sampling_session,
)

pytest.importorskip("ray", reason="Install kumo-core[ray] for backend tests")


def _request(**changes: Any) -> BatchRequest:
    return replace(
        BatchRequest(
            ArtifactIdentity("snapshot", "task", "mapping"),
            ConsumerToken("job", 0, "optimization", 0),
            3,
            0,
            1,
            0,
            (9, 14),
            canonical_json(
                {"stage": "train", "batch_size": 2, "sampling_plan": {"layers": []}}
            ),
            42,
            2,
        ),
        **changes,
    )


def _config() -> ReplicaConfig:
    return ReplicaConfig("runtime.json", "snapshot.json", _request().artifacts, 42, 2)


def _batch(request: BatchRequest) -> dict[str, Any]:
    ids = pa.array(request.seed_ids, type=pa.uint64())
    # Exercise Array slices and separate RecordBatches, including ragged/nulls.
    feature = pa.record_batch(
        {
            "ragged": pa.array([[1, None], None, [], [4]]).slice(1, 2),
            "time": pa.array([None, 2], type=pa.timestamp("ns")),
        }
    )
    return {
        "task": {
            "input_ids": pa.array([0, 1], type=pa.uint32()),
            "tt_stage_node_ids": ids,
            "seed_node_ids": ids,
        },
        "graphs": {
            "main": {
                "node_ids": {"nodes": ids},
                "batch_dict": {"nodes": pa.array([0, 1], type=pa.uint32())},
                "edge_index_dict": [],
                "num_sampled_nodes_dict": {"nodes": pa.array([2], type=pa.uint64())},
                "num_sampled_edges_dict": {},
            }
        },
        "raw_features": {
            "nodes": {
                "num_rows": 2,
                "record_batches": [feature.slice(0, 1), feature.slice(1, 1)],
            }
        },
    }


def test_request_identity_and_envelope_validation() -> None:
    request = _request()
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert len(content_digest(b"manifest")) == 64
    assert request.request_id == _request().request_id
    assert (
        request.request_id
        != _request(consumer=replace(request.consumer, generation=1)).request_id
    )
    payload = BatchPayload(request, _batch(request), 0.1)
    assert payload.validate(request) is payload.batch
    for wrong in (
        replace(request, artifacts=replace(request.artifacts, task_digest="other")),
        replace(request, epoch=4),
    ):
        with pytest.raises(SamplingContractError, match="wrong request"):
            payload.validate(wrong)
    with pytest.raises(SamplingContractError, match="incomplete native"):
        BatchPayload(request, {}, 0).validate(request)
    wrong_batch = _batch(request)
    wrong_batch["task"]["tt_stage_node_ids"] = pa.array([14, 9], type=pa.uint64())
    with pytest.raises(SamplingContractError, match="membership/order"):
        BatchPayload(request, wrong_batch, 0).validate(request)


@pytest.mark.parametrize(
    "changes",
    [
        {"wire_version": 2},
        {"sampling_version": 2},
        {"seed_ids": ()},
        {"seed_ids": (-1,)},
        {"num_threads": 0},
        {"batch_ordinal": -1},
        {"seed_ids": (1, 2, 3)},
        {"options_json": canonical_json({"batch_size": 2, "neg_rhs_node_type": "rhs"})},
    ],
)
def test_request_rejects_invalid_contract(changes: dict[str, Any]) -> None:
    with pytest.raises(SamplingContractError):
        _request(**changes).validate()


def test_shared_capacity_fencing_and_completion() -> None:
    async def exercise() -> None:
        session = _SamplingSession(["a", "b"], _config())
        training = _request()
        rhs = replace(
            training, consumer=replace(training.consumer, consumer_kind="rhs:val")
        )
        slots = [await session.acquire(request, 1) for request in (training, rhs)]
        assert {slot.replica_id for slot in slots} == {0, 1}
        await session.start(slots[0], training)
        await session.start(slots[1], rhs)
        waiting = asyncio.create_task(session.acquire(training, 0.02))
        with pytest.raises(SamplingUnavailableError, match="deadline"):
            await waiting
        await session.fence(training.consumer)
        assert len(session.leases) == 2  # Fencing never releases active scratch.
        with pytest.raises(SamplingContractError, match="obsolete"):
            await session.validate_delivery(training)
        await session.validate_delivery(rhs)
        await session.complete(slots[0].lease_id)
        await session.complete(slots[0].lease_id)  # Idempotent completion.
        third = await session.acquire(rhs, 1)
        assert third.replica_id == slots[0].replica_id
        await session.actor_failed(slots[1].replica_id)
        assert slots[1].lease_id not in session.leases
        await session.close()
        with pytest.raises(SamplingUnavailableError, match="closed"):
            await session.acquire(rhs, 1)

    asyncio.run(exercise())


def test_unstarted_reservation_expiry_and_unavailable_replicas() -> None:
    async def exercise() -> None:
        session = _SamplingSession(["a"], _config())
        request = _request()
        old = await session.acquire(request, 0.001)
        await asyncio.sleep(0.002)
        new = await session.acquire(request, 1)
        with pytest.raises(SamplingContractError, match="reservation"):
            await session.start(old, request)
        await session.start(new, request)
        await session.actor_failed(0)
        with pytest.raises(SamplingUnavailableError, match="no healthy"):
            await session.acquire(request, 1)
        for changed in (
            replace(request, engine_seed=1),
            replace(request, num_threads=1),
            replace(
                request, artifacts=replace(request.artifacts, snapshot_digest="other")
            ),
        ):
            with pytest.raises(SamplingContractError, match="placement"):
                await session.acquire(changed, 1)

    asyncio.run(exercise())


def test_replica_verifies_artifacts_and_budgets() -> None:
    config = _config()
    content = b"immutable manifest"
    digest = content_digest(content)
    config = replace(config, artifacts=ArtifactIdentity(digest, digest, "mapping"))
    with patch("pathlib.Path.read_bytes", return_value=content):
        config.verify()
        with pytest.raises(SamplingContractError, match="digest"):
            replace(config, artifacts=_request().artifacts).verify()
    with pytest.raises(SamplingContractError, match="thread/context"):
        replace(config, num_threads=0).verify()


def test_replica_reuses_context_without_advancing_it() -> None:
    request = _request()
    context = MagicMock()
    context.sample_request.return_value = _batch(request)
    index = MagicMock()
    index.iter.return_value = (context, 2)
    config = _config()
    ray = MagicMock()
    session = MagicMock()
    # Native ownership is tested in the DiskGraph suite; this pins the RPC boundary.
    replica = _SamplerReplica.__new__(_SamplerReplica)
    from collections import OrderedDict

    replica.config, replica.index, replica.contexts = config, index, OrderedDict()
    with patch("kumo_core.diskgraph.ray_sampling._ray", return_value=ray):
        slot = MagicMock()
        one = replica.sample(request, slot, session)
        two = replica.sample(request, slot, session)
    assert one.validate(request)["task"]["tt_stage_node_ids"].equals(
        two.validate(request)["task"]["tt_stage_node_ids"]
    )
    assert index.iter.call_count == 1
    assert context.sample_request.call_count == 2
    context.next_batch.assert_not_called()
    session.complete.remote.assert_not_called()


def test_actor_startup_cleanup_and_explicit_placement() -> None:
    with pytest.raises(SamplingContractError, match="placement"):
        start_sampling_session([], [])
    with pytest.raises(SamplingContractError, match="identical"):
        start_sampling_session(
            [_config(), replace(_config(), engine_seed=2)], ["a", "b"]
        )
    ray = MagicMock()
    ray.get.side_effect = RuntimeError("readiness failed")
    with (
        patch("kumo_core.diskgraph.ray_sampling._ray", return_value=ray),
        pytest.raises(RuntimeError, match="readiness"),
    ):
        start_sampling_session([_config()], ["node-affinity"])
    ray.kill.assert_called_once()
    options = ray.remote.return_value.options.call_args.kwargs
    assert options["max_concurrency"] == 1
    assert options["max_task_retries"] == 0
    assert options["num_cpus"] == 2


def test_shutdown_terminates_actors_even_if_control_plane_fails() -> None:
    ray = MagicMock()
    ray.get.side_effect = RuntimeError("control plane failed")
    with (
        patch("kumo_core.diskgraph.ray_sampling._ray", return_value=ray),
        pytest.raises(RuntimeError, match="control plane"),
    ):
        stop_sampling_session(MagicMock(), ("a", "b"))
    assert ray.kill.call_count == 3


@pytest.fixture(scope="module")
def ray_cluster() -> Any:
    ray = pytest.importorskip("ray")
    assert ray.__version__ == RAY_VERSION
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        object_store_memory=100 * 1024**2,
        _node_ip_address="127.0.0.1",
        namespace="sampling-tests",
    )
    yield ray
    ray.shutdown()


class _ArrowReplica:
    def sample(self, request: BatchRequest, slot: Any, session: Any) -> BatchPayload:
        import ray

        ray.get(session.start.remote(slot, request))
        return BatchPayload(request, _batch(request), 0)


def test_real_ray_array_recordbatch_transport(ray_cluster: Any) -> None:
    ray = ray_cluster
    # Keep the test actor importable in workers regardless of pytest import mode.
    import sys

    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    replicas = [ray.remote(_ArrowReplica).remote() for _ in range(2)]
    session = ray.remote(_SamplingSession).remote(replicas, _config())
    config = RayBackendConfig(
        RayConnection("auto", "sampling-tests"),
        session,
        _request().artifacts,
        42,
        2,
        "job",
    )
    client = RayRuntimeIndex("metadata-only.json", config, _request().consumer)
    try:
        received = client._sample(_request())
        expected = _batch(_request())
        assert received["task"]["seed_node_ids"].equals(
            expected["task"]["seed_node_ids"]
        )
        for actual, original in zip(
            received["raw_features"]["nodes"]["record_batches"],
            expected["raw_features"]["nodes"]["record_batches"],
            strict=True,
        ):
            assert actual.equals(original)
        ray.get(session.fence.remote(_request().consumer))
        with pytest.raises(ray.exceptions.RayTaskError):
            client._sample(_request())
    finally:
        stop_sampling_session(session, tuple(replicas))


class _FailingReplica:
    def sample(self, request: BatchRequest, slot: Any, session: Any) -> BatchPayload:
        import os

        import ray

        ray.get(session.start.remote(slot, request))
        os._exit(1)


class _SlowReplica:
    def sample(self, request: BatchRequest, slot: Any, session: Any) -> BatchPayload:
        import time

        import ray

        ray.get(session.start.remote(slot, request))
        time.sleep(0.5)
        return BatchPayload(request, _batch(request), 0.5)


def test_real_actor_death_replays_the_same_logical_request(ray_cluster: Any) -> None:
    ray = ray_cluster
    import sys

    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    # Admission takes the last free replica first, then retries on the survivor.
    replicas = [
        ray.remote(_ArrowReplica).options(max_restarts=0).remote(),
        ray.remote(_FailingReplica)
        .options(max_restarts=0, max_task_retries=0)
        .remote(),
    ]
    session = ray.remote(_SamplingSession).remote(replicas, _config())
    config = RayBackendConfig(
        RayConnection("auto", "sampling-tests"),
        session,
        _request().artifacts,
        42,
        2,
        "job",
    )
    client = RayRuntimeIndex("metadata-only.json", config, _request().consumer)
    try:
        batch = client._sample(_request())
        assert batch["task"]["tt_stage_node_ids"].to_pylist() == [9, 14]
        assert ray.get(session.stats.remote())["replicas"] == 1
    finally:
        stop_sampling_session(session, tuple(replicas))


def test_real_timeout_fences_delivery_and_preserves_active_slot(
    ray_cluster: Any,
) -> None:
    ray = ray_cluster
    import sys

    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    replicas = [ray.remote(_SlowReplica).remote()]
    session = ray.remote(_SamplingSession).remote(replicas, _config())
    config = RayBackendConfig(
        RayConnection("auto", "sampling-tests"),
        session,
        _request().artifacts,
        42,
        2,
        "job",
        request_timeout=0.05,
    )
    client = RayRuntimeIndex("metadata-only.json", config, _request().consumer)
    try:
        with pytest.raises(ray.exceptions.GetTimeoutError):
            client._sample(_request())
        stats = ray.get(session.stats.remote())
        assert stats["active"] + stats["reserved"] == 1
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(session.validate_delivery.remote(_request()))
        # An unaffected consumer waits for the old actor call to finish.
        other = replace(
            _request(), consumer=replace(_request().consumer, consumer_kind="rhs:val")
        )
        slot = ray.get(session.acquire.remote(other, 5))
        assert slot.replica_id == 0
    finally:
        stop_sampling_session(session, tuple(replicas))


def test_backend_descriptor_delays_actor_handle_deserialization(
    ray_cluster: Any,
) -> None:
    """A new process must connect before deserializing a backend actor handle."""
    import subprocess
    import sys

    ray = ray_cluster
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    replicas = [ray.remote(_ArrowReplica).remote()]
    session = ray.remote(_SamplingSession).remote(replicas, _config())
    config = RayBackendConfig(
        RayConnection("auto", "sampling-tests"),
        session,
        _request().artifacts,
        42,
        2,
        "job",
    )
    assert isinstance(config.session, bytes)
    child = """
import ray
from kumo_core.diskgraph.ray_sampling import RayRuntimeIndex
import sys
assert not ray.is_initialized()
config, request = ray.cloudpickle.loads(sys.stdin.buffer.read())
assert not ray.is_initialized()
client = RayRuntimeIndex('metadata-only.json', config, request.consumer)
batch = client._sample(request)
assert batch['task']['tt_stage_node_ids'].to_pylist() == [9, 14]
ray.shutdown()
"""
    try:
        subprocess.run(
            [sys.executable, "-c", child],
            input=ray.cloudpickle.dumps((config, _request())),
            check=True,
            timeout=30,
            capture_output=True,
        )
    finally:
        stop_sampling_session(session, tuple(replicas))


def test_trainer_rejects_mismatched_seed_manifest_before_scheduling() -> None:
    """A trainer must not select rows from another task's metadata."""
    import os

    client = RayRuntimeIndex.__new__(RayRuntimeIndex)
    client.pid = os.getpid()
    client._manifest_verified = False
    client.manifest_path = "runtime.json"
    client.config = MagicMock()
    client.config.artifacts.task_digest = content_digest(b"expected")
    with (
        patch("pathlib.Path.read_bytes", return_value=b"other task"),
        pytest.raises(SamplingContractError, match="trainer task manifest"),
    ):
        client.iter(stage="train")


def test_request_lookahead_does_not_start_sampling() -> None:
    """Stage quotas can truncate logical requests before any remote submission."""
    import os

    client = RayRuntimeIndex.__new__(RayRuntimeIndex)
    client.pid = os.getpid()
    client._manifest_verified = True
    client.manifest_path = "metadata-only.json"
    client.config = MagicMock()
    client.config.artifacts = _request().artifacts
    client.config.engine_seed = 42
    client.config.num_threads = 2
    client.consumer = _request().consumer
    schedule = MagicMock()
    schedule.next_seed_ids.side_effect = [[9, 14], [17], None]
    with (
        patch(
            "kumo_core.diskgraph.seed_schedule.make_seed_schedule",
            return_value=(schedule, 3),
        ),
        patch.object(client, "_sample") as sample,
    ):
        requests, count = client.iter_requests(
            stage="train", epoch=3, worker_id=0, num_workers=1, batch_size=2
        )
        first, second = list(requests)
        assert count == 3
        assert first.seed_ids == (9, 14) and second.seed_ids == (17,)
        assert first.batch_ordinal == 0 and second.batch_ordinal == 1
        sample.assert_not_called()


@pytest.mark.parametrize("actor_died", [False, True])
def test_cancelled_result_does_not_release_native_capacity_before_actor_drain(
    actor_died: bool,
) -> None:
    """A cancellation-ready ObjectRef is insufficient evidence of native exit."""
    from threading import Event
    from types import SimpleNamespace

    async def exercise() -> None:
        session = _SamplingSession([MagicMock(), MagicMock()], _config())
        training = _request()
        rhs = replace(
            training, consumer=replace(training.consumer, consumer_kind="rhs:val")
        )
        old, other = [await session.acquire(request, 1) for request in (training, rhs)]
        await session.start(old, training)
        await session.start(other, rhs)
        entered, release = Event(), Event()
        barrier = object()
        replica = session.replicas[old.replica_id]
        replica.drain.remote.return_value = barrier
        ray = MagicMock()
        ray.wait.return_value = ([object()], [])
        ray.exceptions = SimpleNamespace(RayActorError=RuntimeError)

        def drain(ref: object) -> None:
            assert ref is barrier
            entered.set()
            assert release.wait(10)
            if actor_died:
                raise RuntimeError("Sampler process died")

        ray.get.side_effect = drain
        with patch("kumo_core.diskgraph.ray_sampling._ray", return_value=ray):
            await session.cancel_consumers([training.consumer])
            watcher = asyncio.create_task(session.watch(old.lease_id, [object()]))
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                assert (await session.stats())["active"] == 2
                assert not session.leases[other.lease_id].cancelling
                await session.validate_delivery(rhs)
                with pytest.raises(SamplingContractError, match="obsolete"):
                    session._validate(training)
                new_token = await session.open_consumer(training.consumer)
                assert new_token.generation == 1
                await session.complete(other.lease_id)
                new_request = replace(training, consumer=new_token)
                replacement = await session.acquire(new_request, 1)
                assert replacement.replica_id != old.replica_id
                await session.start(replacement, new_request)
                # A repeated cancellation of A must not invalidate its replacement.
                await session.cancel_consumers([training.consumer])
                session._validate(new_request)
            finally:
                release.set()
                await watcher
            assert (await session.stats())["active"] == 1
            assert (await session.stats())["available"] == int(not actor_died)
            assert (await session.stats())["replicas"] == (1 if actor_died else 2)

    asyncio.run(exercise())


def test_cancel_releases_reservations_and_wakes_waiting_admission() -> None:
    """An unstarted old call cannot use a slot that has been reassigned."""

    async def exercise() -> None:
        session = _SamplingSession(["sampler"], _config())
        request = _request()
        reserved = await session.acquire(request, 1)
        waiting = asyncio.create_task(session.acquire(request, 10))
        await asyncio.sleep(0)
        await session.cancel_consumers([request.consumer])
        with pytest.raises(SamplingContractError, match="obsolete"):
            await waiting
        with pytest.raises(SamplingContractError, match="obsolete"):
            await session.start(reserved, request)
        assert (await session.stats())["available"] == 1
        token = await session.open_consumer(request.consumer)
        replacement = replace(request, consumer=token)
        slot = await session.acquire(replacement, 1)
        await session.start(slot, replacement)
        assert (await session.stats())["active"] == 1
        await session.close()
        with pytest.raises(SamplingUnavailableError, match="closed"):
            await session.open_consumer(token)

    asyncio.run(exercise())


class _CancellationGate:
    """Coordinate a non-cooperative actor call without timing-dependent sleeps."""

    def __init__(self) -> None:
        self.started_event = asyncio.Event()
        self.release_event = asyncio.Event()

    async def started(self) -> None:
        """Record entry into the simulated native call."""
        self.started_event.set()

    async def wait_started(self) -> None:
        """Wait until native work has entered."""
        await self.started_event.wait()

    async def wait_release(self) -> None:
        """Keep the serial sampler occupied until explicitly released."""
        await self.release_event.wait()

    async def release(self) -> None:
        """Allow native work to finish."""
        self.release_event.set()


class _CancellationReplica(_SamplerReplica):
    """Model a serial Rust call that does not poll Ray's cancellation flag."""

    def __init__(self, gate: Any) -> None:
        self.gate = gate

    def sample(self, request: BatchRequest, slot: Any, session: Any) -> None:
        """Hold native capacity independently of the returned ObjectRef state."""
        import ray

        ray.get(session.start.remote(slot, request))
        ray.get(self.gate.started.remote())
        ray.get(self.gate.wait_release.remote())


def test_real_cancel_waits_for_serial_actor_native_exit(ray_cluster: Any) -> None:
    """The broker preserves a cancelled actor's slot until its work actually exits."""
    import sys

    ray = ray_cluster
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    gate = ray.remote(_CancellationGate).options(num_cpus=0).remote()
    replica = (
        ray.remote(_CancellationReplica)
        .options(num_cpus=1, max_concurrency=1)
        .remote(gate)
    )
    session = (
        ray.remote(_SamplingSession).options(num_cpus=0).remote([replica], _config())
    )
    request = _request()
    try:
        slot = ray.get(session.acquire.remote(request, 10))
        result = replica.sample.remote(request, slot, session)
        completion = session.watch.remote(slot.lease_id, [result])
        ray.get(gate.wait_started.remote(), timeout=10)
        ray.get(session.cancel_consumers.remote([request.consumer]), timeout=10)
        ray.cancel(result, force=False, recursive=False)
        assert not ray.wait([completion], timeout=0.1)[0]
        assert ray.get(session.stats.remote())["active"] == 1
        unrelated = replace(request, consumer=replace(request.consumer, job="other"))
        with pytest.raises(ray.exceptions.RayTaskError, match="deadline"):
            ray.get(session.acquire.remote(unrelated, 0.05), timeout=10)
        ray.get(gate.release.remote(), timeout=10)
        ray.get(completion, timeout=10)
        stats = ray.get(session.stats.remote())
        assert stats["available"] == 1 and stats["active"] == 0
        token = ray.get(session.open_consumer.remote(request.consumer))
        assert token.generation == 1
        replacement = ray.get(
            session.acquire.remote(replace(request, consumer=token), 10)
        )
        assert replacement.replica_id == slot.replica_id
    finally:
        ray.get(gate.release.remote(), timeout=10)
        stop_sampling_session(session, (replica,))
        ray.kill(gate, no_restart=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"seed_ids": [9, 14]},
        {"seed_ids": (True, 14)},
        {"seed_ids": (9.0, 14)},
        {"epoch": True},
        {"options_json": '{"batch_size": 2}'},
        {"options_json": canonical_json({"batch_size": True})},
    ],
)
def test_request_rejects_mutable_or_noninteger_identity(
    changes: dict[str, Any],
) -> None:
    """Reject mutable metadata and values that JSON would coerce ambiguously."""
    with pytest.raises(SamplingContractError):
        _request(**changes).validate()
