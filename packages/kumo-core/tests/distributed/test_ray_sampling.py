# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampling admission, immutable envelopes, and real Ray Arrow serialization."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import replace
from hashlib import sha256
from types import ModuleType
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import diskgraph.runtime_batch as _records
import pyarrow as pa
import pytest
from diskgraph.runtime_batch import RuntimeBatch
from diskgraph.sampling_options import (
    InheritEngineSeed,
    SamplingOptions,
    WeightedSeedSampling,
)
from diskgraph.sampling_plan import FanoutSpec, LayerSpec, SamplingPlan
from kumo_core.diskgraph.batch_contract import (
    WIRE_VERSION,
    ArtifactIdentity,
    BatchRequest,
    HeteroBatchPayload,
    SamplingContractError,
)
from kumo_core.diskgraph.hetero_wire import unpack_python_hetero
from kumo_core.diskgraph.ray_sampling import (
    RayBackendConfig,
    RayRuntimeIndex,
    ReplicaConfig,
    SamplerActor,
    SamplingUnavailableError,
    _ray,
    _SamplerPool,
    _SamplerReplica,
    start_sampling_replicas,
    stop_sampling_replicas,
)

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from torch_geometric.data import HeteroData

pytest.importorskip("ray", reason="Install kumo-core[ray] for backend tests")

_ARTIFACTS = ArtifactIdentity("snapshot", "task", "mapping")


def _request() -> BatchRequest:
    return BatchRequest(
        3,
        0,
        1,
        0,
        (9, 14),
        SamplingOptions(stage="train", batch_size=2),
        42,
        2,
    )


def _invalid_request(changes: dict[str, object]) -> BatchRequest:
    request = _request()
    for name, value in changes.items():
        object.__setattr__(request, name, value)
    return request


def _config() -> ReplicaConfig:
    return ReplicaConfig(
        "runtime.json",
        "snapshot.json",
        _ARTIFACTS,
        42,
        2,
        batch_encoder=_encode_batch,
    )


def _batch(request: BatchRequest) -> RuntimeBatch:
    ids = pa.array(request.seed_ids, type=pa.uint64())
    # Exercise Array slices and separate RecordBatches, including ragged/nulls.
    feature = pa.record_batch(
        {
            "ragged": pa.array([[1, None], None, [], [4]]).slice(1, 2),
            "time": pa.array([None, 2], type=pa.timestamp("ns")),
        }
    )
    return _records.RuntimeBatch(
        task=_records.RuntimeTask(
            input_ids=pa.array([0, 1], type=pa.uint32()),
            tt_stage_node_ids=ids,
            seed_node_ids=ids,
        ),
        graphs={
            "main": _records.RuntimeGraph(
                node_ids={"nodes": ids},
                batch_dict={"nodes": pa.array([0, 1], type=pa.uint32())},
                edge_index_dict=[],
                num_sampled_nodes_dict={"nodes": pa.array([2], type=pa.uint64())},
                num_sampled_edges_dict=[],
            )
        },
        raw_features={
            "nodes": _records.RuntimeFeatureBlock(
                num_rows=2, record_batches=[feature.slice(0, 1), feature.slice(1, 1)]
            )
        },
    )


def _encode_batch(batch: RuntimeBatch) -> HeteroData:
    import torch
    from torch_frame import TensorFrame
    from torch_geometric.data import HeteroData

    ids = batch.task.tt_stage_node_ids
    graph = HeteroData()
    graph["nodes"].n_id = torch.from_numpy(ids.to_numpy().copy()).to(torch.int64)
    graph["nodes"].batch_size = len(ids)
    graph["nodes"].num_sampled_nodes = [len(ids)]
    graph["nodes"].tf = TensorFrame({}, {}, num_rows=len(ids))
    return graph


def _payload(request: BatchRequest) -> HeteroBatchPayload:
    from kumo_core.diskgraph.hetero_wire import pack_python_hetero

    metadata, arrays = pack_python_hetero(_encode_batch(_batch(request)))
    return (metadata, arrays)


def test_response_does_not_echo_request() -> None:
    """Encoded responses carry graph data without duplicating request state."""
    request = _request()
    request.validate()
    payload = _payload(request)
    assert not hasattr(payload, "request")
    assert unpack_python_hetero(*payload)["nodes"].n_id.tolist() == [9, 14]


@pytest.mark.parametrize(
    "changes",
    [
        {"wire_version": WIRE_VERSION + 1},
        {"sampling_version": 2},
        {"seed_ids": ()},
        {"seed_ids": (-1,)},
        {"num_threads": 0},
        {"batch_ordinal": -1},
        {"seed_ids": (1, 2, 3)},
        {
            "options": SamplingOptions(
                stage="train", batch_size=2, neg_rhs_node_type="rhs"
            )
        },
    ],
)
def test_request_rejects_invalid_contract(changes: dict[str, object]) -> None:
    with pytest.raises(SamplingContractError):
        _invalid_request(changes).validate()


def test_replica_verifies_artifacts_and_budgets() -> None:
    config = _config()
    content = b"immutable manifest"
    digest = sha256(content).hexdigest()
    config = replace(config, artifacts=ArtifactIdentity(digest, digest, "mapping"))
    with patch("pathlib.Path.read_bytes", return_value=content):
        config.verify()
        with pytest.raises(SamplingContractError, match="digest"):
            replace(config, artifacts=_ARTIFACTS).verify()
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
    # Native ownership is tested in the DiskGraph suite; this pins the RPC boundary.
    replica = _SamplerReplica.__new__(_SamplerReplica)

    replica.config, replica.index, replica.contexts = config, index, []
    with patch("kumo_core.diskgraph.ray_sampling._ray", return_value=ray):
        one = replica.sample(request)
        two = replica.sample(request)
    assert isinstance(one, tuple) and isinstance(two, tuple)
    assert (
        unpack_python_hetero(*one)["nodes"].n_id.tolist()
        == unpack_python_hetero(*two)["nodes"].n_id.tolist()
    )
    assert index.iter.call_count == 1
    assert context.sample_request.call_count == 2
    context.next_batch.assert_not_called()
    ray.get.assert_not_called()


def test_actor_startup_cleanup_and_explicit_placement() -> None:
    with pytest.raises(SamplingContractError, match="placement"):
        start_sampling_replicas([], [])
    with pytest.raises(SamplingContractError, match="identical"):
        start_sampling_replicas(
            [_config(), replace(_config(), engine_seed=2)], ["a", "b"]
        )
    ray = MagicMock()
    ray.get.side_effect = RuntimeError("readiness failed")
    with (
        patch("kumo_core.diskgraph.ray_sampling._ray", return_value=ray),
        pytest.raises(RuntimeError, match="readiness"),
    ):
        start_sampling_replicas([_config()], ["node-affinity"])
    ray.kill.assert_called_once()
    options = ray.remote.return_value.options.call_args.kwargs
    assert options["max_concurrency"] == 1
    assert options["max_task_retries"] == 0
    assert options["num_cpus"] == 2


def test_ray_import_does_not_require_a_version_or_initialized_runtime() -> None:
    """Dependency discovery leaves runtime compatibility and setup to the caller."""
    module = ModuleType("ray")
    initialize = MagicMock()
    module.__dict__.update({"__version__": "another-version", "init": initialize})
    with patch.dict("sys.modules", {"ray": module}):
        assert _ray() is module
    initialize.assert_not_called()


def test_missing_ray_reports_the_optional_extra() -> None:
    """Explain how to install the backend without making Ray a local dependency."""
    with (
        patch.dict("sys.modules", {"ray": None}),
        pytest.raises(ModuleNotFoundError, match=r"kumo-core\[ray\]") as caught,
    ):
        _ray()
    assert caught.value.name == "ray"


def test_ray_import_preserves_missing_transitive_dependency() -> None:
    """Do not disguise a broken Ray installation as a missing Ray package."""
    error = ModuleNotFoundError("missing Ray dependency", name="other_dependency")
    with (
        patch("builtins.__import__", side_effect=error),
        pytest.raises(ModuleNotFoundError) as caught,
    ):
        _ray()
    assert caught.value is error


@pytest.mark.parametrize("operation", ["client", "start", "stop"])
def test_sampling_requires_caller_initialized_ray(operation: str) -> None:
    """Never let Ray actor/client operations implicitly create a local cluster."""
    ray = MagicMock()
    ray.is_initialized.return_value = False
    actor = cast(SamplerActor, MagicMock())
    with (
        patch("kumo_core.diskgraph.ray_sampling._ray", return_value=ray),
        pytest.raises(RuntimeError, match="Initialize Ray"),
    ):
        if operation == "client":
            _SamplerPool(RayBackendConfig((actor,), _ARTIFACTS, 42, 2))
        elif operation == "start":
            start_sampling_replicas([_config()], ["node-affinity"])
        else:
            stop_sampling_replicas((actor,))
    ray.init.assert_not_called()
    ray.remote.assert_not_called()
    ray.kill.assert_not_called()


def test_empty_replica_cleanup_does_not_load_ray() -> None:
    """Allow launcher cleanup when no sampler actors were created."""
    with patch("kumo_core.diskgraph.ray_sampling._ray") as load:
        stop_sampling_replicas(())
    load.assert_not_called()


def test_backend_config_retains_actor_handles_directly() -> None:
    """Keep handles without importing Ray or serializing them inside configuration."""
    actors = (cast(SamplerActor, MagicMock()),)
    with patch("kumo_core.diskgraph.ray_sampling._ray") as load:
        config = RayBackendConfig(actors, _ARTIFACTS, 42, 2)
    assert config.replicas is actors
    load.assert_not_called()


def test_backend_config_rejects_legacy_encoded_handles() -> None:
    """Reject an opaque descriptor that would require deferred reconnection."""
    with pytest.raises(SamplingContractError, match="nonempty tuple"):
        RayBackendConfig(cast(tuple[SamplerActor, ...], b"actors"), _ARTIFACTS, 42, 2)


@pytest.fixture(scope="module")
def ray_cluster() -> Iterator[ModuleType]:
    ray = pytest.importorskip("ray")
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        object_store_memory=100 * 1024**2,
        _node_ip_address="127.0.0.1",
        namespace="sampling-tests",
    )
    yield ray
    ray.shutdown()


def test_backend_descriptor_uses_caller_initialized_process(
    ray_cluster: ModuleType,
) -> None:
    """The launcher connects each trainer process before passing actor handles."""
    import subprocess
    import sys

    ray = ray_cluster
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    replicas = [ray.remote(_EncodedReplica).remote()]
    config = RayBackendConfig(
        tuple(replicas),
        _ARTIFACTS,
        42,
        2,
    )
    assert config.replicas[0] is replicas[0]
    child = """
import ray
from kumo_core.diskgraph.ray_sampling import RayRuntimeIndex
import sys
assert not ray.is_initialized()
ray.init(address="auto", namespace="sampling-tests")
config, request = ray.cloudpickle.loads(sys.stdin.buffer.read())
assert ray.is_initialized()
client = RayRuntimeIndex('metadata-only.json', config)
batch = client._sample(request)
assert batch["nodes"].n_id.tolist() == [9, 14]
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
        stop_sampling_replicas(tuple(replicas))


def test_trainer_rejects_mismatched_seed_manifest_before_scheduling() -> None:
    """A trainer must not select rows from another task's metadata."""
    import os

    client = RayRuntimeIndex.__new__(RayRuntimeIndex)
    client.pid = os.getpid()
    client._manifest_verified = False
    client.manifest_path = "runtime.json"
    client.config = MagicMock()
    client.config.artifacts.task_digest = sha256(b"expected").hexdigest()
    with (
        patch("pathlib.Path.read_bytes", return_value=b"other task"),
        pytest.raises(SamplingContractError, match="trainer task manifest"),
    ):
        client.iter(SamplingOptions(stage="train", batch_size=2))


@pytest.mark.parametrize("seed", [InheritEngineSeed.DEFAULT, None, 0, 17])
def test_request_lookahead_does_not_start_sampling(
    seed: int | None | InheritEngineSeed,
) -> None:
    """Stage quotas can truncate logical requests before any remote submission."""
    import os

    client = RayRuntimeIndex.__new__(RayRuntimeIndex)
    client.pid = os.getpid()
    client._manifest_verified = True
    client.manifest_path = "metadata-only.json"
    client.config = MagicMock()
    client.config.artifacts = _ARTIFACTS
    client.config.engine_seed = 42
    client.config.num_threads = 2
    schedule = MagicMock()
    schedule.next_seed_ids.side_effect = [[9, 14], [17], None]
    with (
        patch(
            "kumo_core.diskgraph.seed_schedule.make_seed_schedule",
            return_value=(schedule, 3),
        ) as make_schedule,
        patch.object(client, "_sample") as sample,
    ):
        options = SamplingOptions(
            stage="train",
            epoch=3,
            worker_id=1,
            num_workers=2,
            batch_size=2,
            shuffle=True,
            drop_last=True,
            group_by_timestamp=True,
            selected_stage_node_ids=[9, 14, 17],
            weighted_seed_sampling=WeightedSeedSampling(num_samples=3),
            projections={"users": ["age"]},
            rnd_seed=seed,
        )
        requests, count = client.iter_requests(options)
        first, second = list(requests)
        assert count == 3
        assert first.seed_ids == (9, 14) and second.seed_ids == (17,)
        assert first.batch_ordinal == 0 and second.batch_ordinal == 1
        resolved_seed = 42 if seed is InheritEngineSeed.DEFAULT else seed
        make_schedule.assert_called_once_with(
            client.manifest_path, replace(options, rnd_seed=resolved_seed)
        )
        assert first.options == replace(
            options,
            rnd_seed=resolved_seed,
            num_workers=1,
            worker_id=0,
            epoch=0,
            shuffle=False,
            drop_last=False,
            group_by_timestamp=False,
            weighted_seed_sampling=None,
            selected_stage_node_ids=None,
        )
        assert first.options.sampling_plan is options.sampling_plan
        first.validate()
        assert options.rnd_seed is seed
        assert options.selected_stage_node_ids == [9, 14, 17]
        sample.assert_not_called()


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

    def __init__(self, gate: ActorHandle[_CancellationGate]) -> None:
        self.gate = gate

    def sample(self, request: BatchRequest) -> HeteroBatchPayload:
        """Hold native capacity independently of the returned ObjectRef state."""
        import ray

        ray.get(self.gate.started.remote())
        ray.get(self.gate.wait_release.remote())
        return _payload(request)

    def drain(self) -> None:
        """Acknowledge that the serial native call exited."""


@pytest.mark.parametrize(
    "changes",
    [
        {"seed_ids": [9, 14]},
        {"seed_ids": (True, 14)},
        {"seed_ids": (9.0, 14)},
        {"epoch": True},
        {"options": b'{"batch_size": 2}'},
        {"options": SamplingOptions(stage="train", batch_size=True)},
    ],
)
def test_request_rejects_mutable_or_noninteger_identity(
    changes: dict[str, object],
) -> None:
    """Reject mutable metadata and values that JSON would coerce ambiguously."""
    with pytest.raises(SamplingContractError):
        _invalid_request(changes).validate()


class _EncodedReplica:
    """Return encoded graphs with no coordinator calls."""

    def sample(self, request: BatchRequest) -> HeteroBatchPayload:
        """Return the requested dictionary/NumPy tuple."""
        return _payload(request)

    def drain(self) -> None:
        """Acknowledge prior serial calls."""


class _FailingReplica(_EncodedReplica):
    def sample(self, request: BatchRequest) -> HeteroBatchPayload:
        """Simulate confirmed actor death."""
        import os

        os._exit(1)


def _backend(
    replicas: tuple[SamplerActor, ...], request_timeout: float = 300.0
) -> RayBackendConfig:
    return RayBackendConfig(
        replicas,
        _ARTIFACTS,
        42,
        2,
        request_timeout=request_timeout,
    )


def _actors(ray: ModuleType, count: int = 2) -> tuple[SamplerActor, ...]:
    import sys

    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    return tuple(
        cast(SamplerActor, ray.remote(_EncodedReplica).remote()) for _ in range(count)
    )


def test_real_actor_tracking_and_encoded_delivery(ray_cluster: ModuleType) -> None:
    """Keep one request per actor and refill the actor whose result was consumed."""
    ray = ray_cluster
    replicas = _actors(ray)
    pool = _SamplerPool(_backend(replicas))
    try:
        one, two = (pool.submit(_request()) for _ in range(2))
        assert len(set(pool.pending.values())) == 2
        with pytest.raises(SamplingUnavailableError, match="consume a result"):
            pool.submit(_request())
        second_actor = pool.pending[two]
        received = unpack_python_hetero(*cast(HeteroBatchPayload, pool.get(two)))
        assert received["nodes"].n_id.tolist() == [9, 14]
        three = pool.submit(_request())
        assert pool.pending[three] == second_actor
        unpack_python_hetero(*cast(HeteroBatchPayload, pool.get(one)))
        unpack_python_hetero(*cast(HeteroBatchPayload, pool.get(three)))
        assert not pool.pending and len(pool.available) == 2
    finally:
        pool.close()
        stop_sampling_replicas(replicas)


def test_real_actor_death_retries_on_survivor(ray_cluster: ModuleType) -> None:
    """Retry the identical logical request after a terminal actor failure."""
    ray = ray_cluster
    survivor = _actors(ray, 1)[0]
    failed = (
        ray.remote(_FailingReplica).options(max_restarts=0, max_task_retries=0).remote()
    )
    replicas = failed, survivor
    client = RayRuntimeIndex("metadata-only.json", _backend(replicas))
    try:
        batch = client._sample(_request())
        assert batch["nodes"].n_id.tolist() == [9, 14]
        assert list(client.samplers.available) == [1]
    finally:
        client.close()
        stop_sampling_replicas(replicas)


def test_direct_call_has_no_control_rpcs() -> None:
    """Successful delivery issues only sample and result retrieval calls."""
    ray = MagicMock()
    actor = MagicMock()
    ref = actor.sample.remote.return_value
    ray.get.return_value = _payload(_request())
    config = RayBackendConfig((actor,), _ARTIFACTS, 42, 2)
    with patch("kumo_core.diskgraph.ray_sampling._ray", return_value=ray):
        pool = _SamplerPool(config)
        unpack_python_hetero(
            *cast(HeteroBatchPayload, pool.get(pool.submit(_request())))
        )
        pool.close()
    actor.sample.remote.assert_called_once_with(_request())
    ray.get.assert_called_once_with(ref, timeout=config.request_timeout)
    actor.drain.remote.assert_not_called()
    ray.cancel.assert_not_called()


def test_shutdown_kills_only_owned_actors() -> None:
    """Shutdown needs no broker or Ray runtime shutdown."""
    ray = MagicMock()
    with patch("kumo_core.diskgraph.ray_sampling._ray", return_value=ray):
        stop_sampling_replicas((MagicMock(), MagicMock()))
    assert ray.kill.call_count == 2
    ray.shutdown.assert_not_called()
    ray.get.assert_not_called()


@pytest.mark.parametrize("timeout_first", [False, True])
def test_cancellation_drains_native_work_before_reuse(
    ray_cluster: ModuleType, timeout_first: bool
) -> None:
    """Cancellation-ready refs retain actor ownership until a serial drain finishes."""
    import sys
    from concurrent.futures import ThreadPoolExecutor

    ray = ray_cluster
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    gate = ray.remote(_CancellationGate).options(num_cpus=0).remote()
    replica = (
        ray.remote(_CancellationReplica)
        .options(num_cpus=1, max_concurrency=1)
        .remote(gate)
    )
    pool = _SamplerPool(
        _backend((replica,), request_timeout=0.05 if timeout_first else 10)
    )
    result = pool.submit(_request())
    try:
        ray.get(gate.wait_started.remote(), timeout=10)
        if timeout_first:
            with pytest.raises(ray.exceptions.GetTimeoutError):
                pool.get(result)
            assert result in pool.pending and not pool.available
            # Even the drain timeout must retain ownership so close can be retried.
            with pytest.raises(ray.exceptions.GetTimeoutError):
                pool.close()
            assert result in pool.pending
            pool.config = replace(pool.config, request_timeout=10)
        pool.cancel()
        with pytest.raises(SamplingUnavailableError, match="closed"):
            pool.submit(_request())
        with ThreadPoolExecutor(max_workers=1) as executor:
            closing = executor.submit(pool.close)
            try:
                with pytest.raises(TimeoutError):
                    closing.result(timeout=0.1)
                assert result in pool.pending
            finally:
                ray.get(gate.release.remote(), timeout=10)
            closing.result(timeout=10)
        assert not pool.pending
        with pytest.raises(SamplingUnavailableError):
            pool.get(result)
        restarted = _SamplerPool(_backend((replica,)))
        unpack_python_hetero(
            *cast(HeteroBatchPayload, restarted.get(restarted.submit(_request())))
        )
        restarted.close()
    finally:
        ray.get(gate.release.remote(), timeout=10)
        stop_sampling_replicas((replica,))
        ray.kill(gate, no_restart=True)


def test_replica_reuses_equal_native_plans_and_evicts_old_contexts() -> None:
    """Structural plan equality drives the bounded LRU without serialized keys."""
    request = _request()
    context = MagicMock()
    context.sample_request.return_value = _batch(request)
    index = MagicMock()
    index.iter.return_value = (context, 2)
    replica = _SamplerReplica.__new__(_SamplerReplica)
    replica.config = replace(_config(), context_capacity=2)
    replica.index, replica.contexts = index, []
    replica.sample(request)
    replica.sample(
        replace(request, options=replace(request.options, sampling_plan=SamplingPlan()))
    )
    assert index.iter.call_count == 1
    second = replace(
        request.options,
        sampling_plan=SamplingPlan(
            [LayerSpec(default_fanout=FanoutSpec(4, "uniform", "reject_plan"))]
        ),
    )
    third = replace(
        request.options,
        sampling_plan=SamplingPlan(
            [LayerSpec(default_fanout=FanoutSpec(8, "uniform", "reject_plan"))]
        ),
    )
    replica.sample(replace(request, options=second))
    replica.sample(request)
    replica.sample(replace(request, options=third))
    assert index.iter.call_count == 3
    assert [options for options, _ in replica.contexts] == [request.options, third]


def test_replica_rejects_missing_encoder() -> None:
    """Actor startup must not silently select raw delivery."""
    config = _config()
    object.__setattr__(config, "batch_encoder", None)
    with pytest.raises(SamplingContractError, match="encoder"):
        config.verify()


def test_replica_rejects_wrong_sample_ids_before_encoding() -> None:
    """Validate native seed membership before invoking the configured encoder."""
    request = _request()
    replica = _SamplerReplica.__new__(_SamplerReplica)
    encoder = MagicMock()
    replica.config = replace(_config(), batch_encoder=encoder)
    replica.index = MagicMock()
    replica.contexts = []
    context = MagicMock()
    context.sample_request.return_value = _batch(replace(request, seed_ids=(14, 9)))
    replica.index.iter.return_value = context, 2
    with pytest.raises(SamplingContractError, match="membership/order"):
        replica.sample(request)
    encoder.assert_not_called()
