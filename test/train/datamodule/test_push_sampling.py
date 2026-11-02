# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native shard streaming over prepared worker-local task artifacts."""

from __future__ import annotations

import shutil
import time
from collections import deque
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from test.train.datamodule import test_remote_sampling_integration as parity
from typing import Any

import diskgraph
import orjson
import pyarrow as pa
import pytest
from kumo_core.diskgraph.lightning_data import (
    DiskgraphLightningDataModule,
    DiskgraphTaskManifest,
)
from kumo_core.diskgraph.sampling_service import (
    SamplingService,
    SamplingServiceConfig,
)
from pyarrow import flight

prepared_task = parity.prepared_task


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    """Keep synthetic sampling tests independent of application services."""
    return False


def _options() -> dict[str, Any]:
    return {
        "stage": "train",
        "sampling_plan": {"layers": []},
        "batch_size": 2,
        "num_workers": 1,
        "worker_id": 0,
        "shuffle": False,
        "epoch": 0,
        "entity_node_type": "users",
        "fused": False,
    }


def _wait_for_idle(service: SamplingService) -> None:
    deadline = time.monotonic() + 5
    while service.health()["active_streams"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert service.health()["active_streams"] == 0


@pytest.mark.parametrize("replicas", [1, 2, 3, 8])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"shuffle": True, "epoch": 5},
        {"drop_last": True},
        {"selected_stage_node_ids": [5, 1, 3]},
        {"num_workers": 2, "worker_id": 1},
        {"projections": {"users": ["age"], "orders": ["amount"]}},
    ],
)
def test_push_matches_native_shards(
    prepared_task: str, tmp_path: Path, replicas: int, options: dict[str, Any]
) -> None:
    """Complete remote batches match the same native shards, including empty ones."""
    source = Path(prepared_task).parent / "graph"
    with ExitStack() as cleanup:
        services = []
        for replica in range(replicas):
            root = tmp_path / f"replica-{replica}"
            shutil.copytree(source, root)
            service = SamplingService(SamplingServiceConfig(root))
            services.append(service)
            cleanup.callback(service.close)
        runtime = diskgraph.RuntimeIndex(
            prepared_task,
            sampling_endpoints=[service.endpoint() for service in services],
            rnd_seed=42,
            num_threads=1,
        )
        cleanup.callback(runtime.close)
        local = diskgraph.RuntimeIndex(prepared_task, rnd_seed=42, num_threads=1)
        kwargs = {**_options(), **options}
        expected = []
        count = 0
        shards: deque[Iterator[dict[str, Any]]] = deque()
        for replica in range(replicas):
            shard, size = local.iter(
                **{
                    **kwargs,
                    "num_workers": kwargs["num_workers"] * replicas,
                    "worker_id": kwargs["worker_id"] + kwargs["num_workers"] * replica,
                }
            )
            count += size
            shards.append(shard)
        while shards:
            shard = shards.popleft()
            try:
                expected.append(next(shard))
            except StopIteration:
                continue
            shards.append(shard)
        actual, size = runtime.iter(**kwargs)
        assert size == count
        parity._assert_equal(expected, list(actual))
        assert all(service.health()["task_requests"] == 1 for service in services)
        assert all(service.health()["get_streams"] == 1 for service in services)


def _wait_for_backpressure(service: SamplingService, total: int) -> int:
    deadline = time.monotonic() + 10
    previous = -1
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        current = int(service.health()["batches"])
        assert current < total, "Paused reader allowed the full epoch to be generated"
        if current != previous:
            previous = current
            stable_since = time.monotonic()
        elif current > 0 and time.monotonic() - stable_since >= 0.3:
            return current
        time.sleep(0.02)
    raise AssertionError("Native production did not stop under transport backpressure")


@pytest.mark.parametrize("shutdown", [False, True])
def test_transport_backpressure_stops_and_resumes_native_production(
    prepared_task: str, shutdown: bool
) -> None:
    """A long epoch stalls without reader progress, resumes, and cancels cleanly."""
    service = SamplingService(
        SamplingServiceConfig(Path(prepared_task).parent / "graph")
    )
    runtime = diskgraph.RuntimeIndex(
        prepared_task, sampling_endpoints=service.endpoint(), rnd_seed=42, num_threads=1
    )
    total = 100_000
    try:
        iterator, count = runtime.iter(
            **{**_options(), "batch_size": 1, "selected_stage_node_ids": [0] * total}
        )
        assert count == total
        paused = _wait_for_backpressure(service, total)
        # Drain existing prefetch and force at least one additional native batch.
        for _ in range(paused + 1):
            next(iterator)
        deadline = time.monotonic() + 5
        while (
            int(service.health()["batches"]) <= paused and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert int(service.health()["batches"]) > paused
        _wait_for_backpressure(service, total)
        if shutdown:
            started = time.monotonic()
            service.close()
            assert service.health()["ready"] is False
            assert service.health()["active_streams"] == 0
            # A paused transport may use the listener's five-second drain grace.
            assert time.monotonic() - started < 8
        else:
            iterator.close()
            _wait_for_idle(service)
            complete, _ = runtime.iter(**_options())
            assert len(list(complete)) == 3
            _wait_for_idle(service)
    finally:
        runtime.close()
        service.close()


def test_prepared_task_is_reused_without_artifact_rpcs(
    prepared_task: str, tmp_path: Path
) -> None:
    """Prepared artifacts serve repeated epochs without registration or uploads."""
    root = tmp_path / "replica"
    shutil.copytree(Path(prepared_task).parent / "graph", root)
    trainer_root = tmp_path / "trainer-only"
    manifest = orjson.loads(Path(prepared_task).read_bytes())
    manifest["graph"]["index_path"] = str(trainer_root)
    manifest["stages"]["train"]["index_path"] = str(
        trainer_root / "train_tables/remote_test"
    )
    portable_manifest = tmp_path / "portable.json"
    portable_manifest.write_bytes(orjson.dumps(manifest))
    service = SamplingService(SamplingServiceConfig(root))
    runtime = diskgraph.RuntimeIndex(
        str(portable_manifest),
        sampling_endpoints=service.endpoint(),
        rnd_seed=42,
        num_threads=1,
    )
    try:
        for epoch in range(3):
            iterator, _ = runtime.iter(**{**_options(), "epoch": epoch})
            assert len(list(iterator)) == 3
        assert service.health()["get_streams"] == 3
        assert not trainer_root.exists()
    finally:
        runtime.close()
        service.close()


@pytest.mark.parametrize("incomplete", [False, True])
def test_missing_task_artifact_fails_until_materialized(
    prepared_task: str, tmp_path: Path, incomplete: bool
) -> None:
    """The sampler rejects missing artifacts and resumes after external preparation."""
    root = tmp_path / "replica"
    source = Path(prepared_task).parent / "graph"
    shutil.copytree(source, root, ignore=shutil.ignore_patterns("train_tables"))
    extension = root / "train_tables/remote_test"
    if incomplete:
        extension.mkdir(parents=True)
    service = SamplingService(SamplingServiceConfig(root))
    runtime = diskgraph.RuntimeIndex(
        prepared_task, sampling_endpoints=service.endpoint(), rnd_seed=42, num_threads=1
    )
    try:
        with pytest.raises(
            (flight.FlightServerError, pa.ArrowInvalid), match="materialize.*worker"
        ):
            runtime.iter(**_options())
        assert not (extension / "index_metadata.json").exists()
        _wait_for_idle(service)
        shutil.copytree(
            source / "train_tables/remote_test", extension, dirs_exist_ok=True
        )
        iterator, _ = runtime.iter(**_options())
        assert len(list(iterator)) == 3
    finally:
        runtime.close()
        service.close()


@pytest.mark.parametrize("replicas", [1, 2, 3, 8])
@pytest.mark.parametrize("drop_last", [False, True])
def test_remote_loader_length_matches_replica_batch_tails(
    prepared_task: str, tmp_path: Path, replicas: int, drop_last: bool
) -> None:
    """Count every replica's partial tail before Lightning limits an epoch."""
    from kumo_core.diskgraph.lightning_data import (
        DiskgraphLightningDataModule,
        DiskgraphTaskBatchIterable,
        _ShuffleStreamState,
    )

    graph = Path(prepared_task).parent / "graph"
    with ExitStack() as cleanup:
        endpoints = []
        for replica in range(replicas):
            root = tmp_path / f"replica-{replica}"
            shutil.copytree(graph, root)
            service = SamplingService(SamplingServiceConfig(root))
            cleanup.callback(service.close)
            endpoints.append(service.endpoint())
        options = {
            "remote_sampling_endpoint": endpoints,
            "entity_node_type": "users",
            "fused": False,
        }
        dataset = DiskgraphTaskBatchIterable(
            manifest_path=prepared_task,
            stage="train",
            sampling_plan={"layers": []},
            batch_size=2,
            shuffle=False,
            drop_last=drop_last,
            base_seed=42,
            shuffle_state=_ShuffleStreamState(),
            num_seeds=6,
            runtime_options=options,
        )
        expected = len(dataset)
        assert len(list(dataset)) == expected
        assert len(list(dataset)) == expected
        module = object.__new__(DiskgraphLightningDataModule)
        module.runtime_options = options
        assert module._natural_batches(6, 2, drop_last=drop_last) == expected


def test_ordered_rhs_catalog_scan_uses_one_replica(prepared_task: str) -> None:
    """MIPS embeddings retain the local iterator's global node order."""
    graph = Path(prepared_task).parent / "graph"
    with ExitStack() as cleanup:
        services = [SamplingService(SamplingServiceConfig(graph)) for _ in range(3)]
        for service in services:
            cleanup.callback(service.close)
        remote = diskgraph.RuntimeIndex(
            prepared_task,
            sampling_endpoints=[s.endpoint() for s in services],
            rnd_seed=42,
            num_threads=1,
        )
        cleanup.callback(remote.close)
        options = {**_options(), "rhs_seed_node_type": "users", "rhs_seed_time": 0}
        expected, count = diskgraph.RuntimeIndex(prepared_task, 42, 1).iter(**options)
        actual, remote_count = remote.iter(**options)
        assert count == remote_count == 8
        parity._assert_equal(list(expected), list(actual))
        assert [s.health()["task_requests"] for s in services] == [1, 0, 0]


def test_replicas_produce_concurrently(prepared_task: str) -> None:
    """Independent native listeners serve epochs without coordinating production."""
    graph = Path(prepared_task).parent / "graph"
    with ExitStack() as cleanup:
        first = SamplingService(SamplingServiceConfig(graph))
        second = SamplingService(SamplingServiceConfig(graph))
        cleanup.callback(first.close)
        cleanup.callback(second.close)
        client = diskgraph.SamplingClient(first.endpoint())
        cleanup.callback(client.close)
        request = diskgraph.SamplingRequest(
            "train_tables/remote_test",
            engine_seed=42,
            num_threads=1,
            stage_table="train_seeds",
            sampling_plan_json='{"layers":[]}',
            batch_size=2,
            entity_node_type="users",
            fused=False,
        )
        reader = client.open(request)
        assert (reader.num_samples, reader.num_batches) == (6, 3)
        assert set(next(reader)) == {"task", "graphs", "raw_features"}
        runtime = diskgraph.RuntimeIndex(
            prepared_task,
            sampling_endpoints=second.endpoint(),
            rnd_seed=42,
            num_threads=1,
        )
        cleanup.callback(runtime.close)
        try:
            iterator, _ = runtime.iter(**_options())
            assert len(list(iterator)) == 3
            assert first.health()["task_requests"] == 1
            assert second.health()["batches"] == 3
        finally:
            reader.close()


def test_native_request_supports_numpy_seed_selection(prepared_task: str) -> None:
    """Native extraction and ticket serialization preserve NumPy-selected seeds."""
    import numpy as np

    graph = Path(prepared_task).parent / "graph"
    with ExitStack() as cleanup:
        service = SamplingService(SamplingServiceConfig(graph))
        cleanup.callback(service.close)
        client = diskgraph.SamplingClient(service.endpoint())
        cleanup.callback(client.close)
        request = diskgraph.SamplingRequest(
            "train_tables/remote_test",
            stage_table="train_seeds",
            sampling_plan_json='{"layers":[]}',
            batch_size=2,
            engine_seed=np.uint64(42),
            selected_stage_node_ids=np.array([1, 3, 5], dtype=np.uint64),
            entity_node_type="users",
            fused=False,
        )
        stream = client.open(request)
        cleanup.callback(stream.close)
        assert (stream.num_samples, stream.num_batches) == (3, 2)
        batches = list(stream)
        assert [
            value
            for batch in batches
            for value in batch["task"]["tt_stage_node_ids"].to_pylist()
        ] == [1, 3, 5]


def test_remote_backend_drop_cancels_a_live_epoch(prepared_task: str) -> None:
    """Dropping the native owner cancels paused readers without joining under the GIL."""
    import gc

    graph = Path(prepared_task).parent / "graph"
    with ExitStack() as cleanup:
        service = SamplingService(SamplingServiceConfig(graph))
        cleanup.callback(service.close)
        backend = diskgraph.RemoteRuntimeIndex([service.endpoint()])
        request = diskgraph.SamplingRequest(
            "train_tables/remote_test",
            stage_table="train_seeds",
            sampling_plan_json='{"layers":[]}',
            batch_size=1,
            entity_node_type="users",
            fused=False,
            selected_stage_node_ids=[0] * 10000,
        )
        iterator = backend.iter(request, str(graph))
        cleanup.callback(iterator.close)
        next(iterator)
        del backend
        gc.collect()
        _wait_for_idle(service)
        assert list(iterator) == []


@pytest.mark.parametrize("native_server", [False, True])
@pytest.mark.parametrize("stage_count", [2, 3])
@pytest.mark.parametrize("worker_count", [1, 2])
def test_refit_stream_budget_allows_paused_stage_streams(
    prepared_task: str,
    tmp_path: Path,
    native_server: bool,
    stage_count: int,
    worker_count: int,
) -> None:
    """Open every worker's stage streams while earlier producers are paused."""
    graph = Path(prepared_task).parent / "graph"
    stages = ("train", "val", "test")[:stage_count]
    manifest = orjson.loads(Path(prepared_task).read_bytes())
    manifest["stages"] = {stage: manifest["stages"]["train"] for stage in stages}
    manifest_path = tmp_path / "refit_manifest.json"
    manifest_path.write_bytes(orjson.dumps(manifest))
    stream_count = stage_count * worker_count
    options = {"max_streams": stream_count} if worker_count > 1 else {}
    config = (
        SamplingServiceConfig(graph)
        if worker_count == 1
        else SamplingServiceConfig(graph, max_streams=stream_count)
    )
    with ExitStack() as cleanup:
        service = (
            diskgraph.SamplingServer(graph, **options)
            if native_server
            else SamplingService(config)
        )
        cleanup.callback(service.close)
        runtime = diskgraph.RuntimeIndex(
            str(manifest_path),
            sampling_endpoints=service.endpoint(),
            rnd_seed=42,
            num_threads=1,
        )
        cleanup.callback(runtime.close)
        streams = []
        for worker_id in range(worker_count):
            for stage in stages:
                stream, _ = runtime.iter(
                    **{
                        **_options(),
                        "stage": stage,
                        "batch_size": 1,
                        "num_workers": worker_count,
                        "worker_id": worker_id,
                        "selected_stage_node_ids": [0] * 10000,
                    }
                )
                cleanup.callback(stream.close)
                streams.append(stream)
                next(stream)

        assert service.health()["active_streams"] == stream_count
        for stream in streams:
            assert next(stream)["task"]["input_ids"].to_pylist() == [0]
        runtime.close()
        _wait_for_idle(service)


@pytest.mark.parametrize(
    ("batch_size", "drop_last", "expected_batches"),
    [(3, True, 2), (4, True, 0), (4, False, 2)],
)
def test_remote_training_ignores_requested_loader_processes(
    prepared_task: str,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    drop_last: bool,
    expected_batches: int,
) -> None:
    """Native replica batches match loader length without extra worker sharding."""
    graph = Path(prepared_task).parent / "graph"
    monkeypatch.setattr(
        DiskgraphTaskManifest, "stage_counts", property(lambda self: {"train": 6})
    )
    manifest = DiskgraphTaskManifest(
        manifest_path=prepared_task,
        runtime_manifest_path=prepared_task,
        manifest={},
        metadata={"train_table_name": "train_seeds", "seed_node_type": "users"},
        feature_schema={},
        graph_runtime_metadata={},
    )
    with ExitStack() as cleanup:
        services = [SamplingService(SamplingServiceConfig(graph)) for _ in range(2)]
        for service in services:
            cleanup.callback(service.close)
        dm = DiskgraphLightningDataModule(
            manifest,
            {},
            sampling_plan={"layers": []},
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last_train=drop_last,
            resolve_entity_node_type="users",
            runtime_options={
                "remote_sampling_endpoint": [
                    service.endpoint() for service in services
                ],
                "fused": False,
            },
        )
        cleanup.callback(dm._runtime_owner.close)
        monkeypatch.setattr(dm, "_to_hetero_train", lambda batch: batch)
        if not expected_batches:
            with pytest.raises(ValueError, match="no native batches"):
                dm.train_dataloader()
            assert all(service.health()["task_requests"] == 0 for service in services)
            return

        loader = dm.train_dataloader()
        assert loader.num_workers == 0
        batches = list(loader)
        assert len(loader) == len(batches) == expected_batches
        assert sorted(
            value
            for batch in batches
            for value in batch["task"]["tt_stage_node_ids"].to_pylist()
        ) == list(range(6))
        assert dm.active_dataloader_worker_pids == ()


def test_shared_native_backend_closes_all_active_epochs(prepared_task: str) -> None:
    """One manifest wrapper owns multiple native epoch streams and cancels them together."""
    graph = Path(prepared_task).parent / "graph"
    with ExitStack() as cleanup:
        service = SamplingService(SamplingServiceConfig(graph, max_streams=2))
        cleanup.callback(service.close)
        runtime = diskgraph.RuntimeIndex(
            prepared_task,
            sampling_endpoints=service.endpoint(),
            rnd_seed=42,
            num_threads=1,
        )
        cleanup.callback(runtime.close)
        streams = [
            runtime.iter(
                **{
                    **_options(),
                    "batch_size": 1,
                    "selected_stage_node_ids": [0] * 10000,
                }
            )[0]
            for _ in range(2)
        ]
        for stream in streams:
            next(stream)
        streams[0].close()
        assert next(streams[1])["task"]["input_ids"].to_pylist() == [0]
        replacement, _ = runtime.iter(
            **{**_options(), "batch_size": 1, "selected_stage_node_ids": [0] * 10000}
        )
        streams.append(replacement)
        next(replacement)
        runtime.close()
        _wait_for_idle(service)
        for stream in streams:
            assert list(stream) == []
