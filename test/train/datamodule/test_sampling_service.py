# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real native iterator parity and native gateway lifecycle checks."""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from test.train.datamodule import test_remote_sampling_integration as parity

import orjson
import pyarrow as pa
import pytest
from kumo_core.diskgraph.remote_sampling import RemoteTaskIterator
from kumo_core.diskgraph.sampling_service import (
    SamplingService,
    SamplingServiceConfig,
)
from pyarrow import flight

prepared_task = parity.prepared_task
test_local_remote_parity = parity.test_local_remote_parity
test_cancellation_releases_gateway_streams = (
    parity.test_cancellation_releases_gateway_streams
)


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    """Use local artifacts without application services."""
    return False


@pytest.fixture(scope="module")
def remote_task(prepared_task: str) -> Iterator[tuple[str, str]]:
    """Serve real task iterators through the embedded native listener."""
    source = Path(prepared_task).parent / "graph"
    root = source.parent / "sampler_graph"
    shutil.copytree(
        source, root, ignore=shutil.ignore_patterns("train_tables", "pred_tables")
    )
    assert not (root / "train_tables").exists()
    service = SamplingService(SamplingServiceConfig(root))
    try:
        assert service.health()["pid"] == os.getpid()
        yield prepared_task, service.endpoint()
        deadline = time.monotonic() + 5
        while service.health()["active_streams"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.health()["active_streams"] == 0
        assert int(service.health()["batches"]) > 0
    finally:
        service.close()
    assert service.health()["ready"] is False


@pytest.mark.parametrize("relative", ["../escape", "/tmp", "train_tables/../.."])
def test_service_rejects_unsafe_paths(prepared_task: str, relative: str) -> None:
    """A malformed request must not escape the actor's immutable index root."""
    service = SamplingService(
        SamplingServiceConfig(Path(prepared_task).parent / "graph")
    )
    try:
        request = {"kind": "push_iter", "version": 4, "extension_ref": relative}
        with pytest.raises((flight.FlightServerError, pa.ArrowInvalid)):
            RemoteTaskIterator(service.endpoint(), request)
        assert service.health()["active_streams"] == 0
    finally:
        service.close()


def test_health_action(remote_task: tuple[str, str]) -> None:
    """A Flight health request exposes the owner PID without consuming a stream."""
    _, endpoint = remote_task
    with flight.connect(endpoint) as client:
        results = list(client.do_action(flight.Action("health", b"")))
    health = orjson.loads(results[0].body.to_pybytes())
    assert health["ready"] is True
    assert health["pid"] == os.getpid()


@pytest.mark.parametrize("replica_count", [1, 3])
def test_epoch_shards_use_all_replicas(
    prepared_task: str, tmp_path: Path, replica_count: int
) -> None:
    """Real Flight replicas share batches evenly and preserve local batch contents."""
    from contextlib import ExitStack

    import diskgraph
    from kumo_core.diskgraph.remote_sampling import RemoteRuntimeIndex

    graph = tmp_path / "graph"
    shutil.copytree(
        Path(prepared_task).parent / "graph",
        graph,
        ignore=shutil.ignore_patterns("train_tables", "pred_tables"),
    )
    with ExitStack() as cleanup:
        services = []
        for _ in range(replica_count):
            service = SamplingService(SamplingServiceConfig(graph))
            cleanup.callback(service.close)
            services.append(service)
        remote = RemoteRuntimeIndex(
            prepared_task,
            [service.endpoint() for service in services],
            rnd_seed=42,
            num_threads=1,
        )
        cleanup.callback(remote.close)
        local = diskgraph.RuntimeIndex(prepared_task, rnd_seed=42, num_threads=1)
        options = {
            "stage": "train",
            "entity_node_type": "users",
            "batch_size": 1,
            "num_workers": 1,
            "worker_id": 0,
            "epoch": 0,
            "sampling_plan": {
                "layers": [
                    {
                        "fanouts": {},
                        "default_fanout": {
                            "fanout": 100,
                            "mode": "uniform",
                            "absent_key_policy": "uniform_fallback",
                        },
                    }
                ]
            },
            "shuffle": False,
            "fused": False,
        }
        expected, expected_count = local.iter(**options)
        actual, actual_count = remote.iter(**options)
        assert actual_count == expected_count == 6
        parity._assert_equal(list(expected), list(actual))
        assert [service.health()["batches"] for service in services] == [
            6 // replica_count
        ] * replica_count
        assert all(service.health()["task_requests"] == 1 for service in services)
        assert (graph / "train_tables" / "remote_test").is_dir()
        clients = list(remote._clients.values())
        assert len(clients) == replica_count
        remote.close()
        assert not remote._clients


@pytest.mark.parametrize("seed", [-1, 2**64, True])
def test_service_rejects_invalid_engine_seeds(prepared_task: str, seed: int) -> None:
    """Reject seeds that the wire format or native unsigned API cannot represent."""
    service = SamplingService(
        SamplingServiceConfig(Path(prepared_task).parent / "graph")
    )
    request = {
        "kind": "push_iter",
        "version": 4,
        "extension_ref": "train_tables/remote_test",
        "engine_seed": seed,
    }
    try:
        expected = (
            TypeError if seed == 2**64 else (flight.FlightServerError, pa.ArrowInvalid)
        )
        message = "64-bit range" if seed == 2**64 else "u64|unsigned 64-bit"
        with pytest.raises(expected, match=message):
            RemoteTaskIterator(service.endpoint(), request)
        assert service.health()["active_streams"] == 0
    finally:
        service.close()


def test_native_progress_logs_reach_python(
    prepared_task: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Native lifecycle and progress events reach the actor's Python logger."""
    from kumo_core.diskgraph.remote_sampling import RemoteRuntimeIndex

    caplog.set_level("INFO")
    service = SamplingService(
        SamplingServiceConfig(Path(prepared_task).parent / "graph")
    )
    runtime = RemoteRuntimeIndex(prepared_task, service.endpoint(), 42, 1)
    try:
        iterator, _ = runtime.iter(
            stage="train",
            sampling_plan={"layers": []},
            num_workers=1,
            worker_id=0,
            shuffle=False,
            epoch=0,
            batch_size=2,
            entity_node_type="users",
            fused=False,
        )
        assert len(list(iterator)) == 3
        assert service.health()["batches"] == 3
    finally:
        runtime.close()
        service.close()
    log = caplog.text
    for message in (
        "SAMPLING_LISTENER_READY",
        "SAMPLING_PROGRESS",
        "SAMPLING_STREAM_CLOSED",
        "SAMPLING_LISTENER_STOPPED",
    ):
        assert message in log
    assert "batches=1" in log
