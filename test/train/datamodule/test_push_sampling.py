# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native feeder sharding, Flight credits and immutable task transfer."""

from __future__ import annotations

import shutil
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from test.train.datamodule import test_remote_sampling_integration as parity
from typing import Any
from unittest.mock import MagicMock

import diskgraph
import orjson
import pyarrow as pa
import pytest
from kumo_core.diskgraph.remote_sampling import (
    RemoteRuntimeIndex,
    RemoteTaskIterator,
)
from kumo_core.diskgraph.sampling_protocol import (
    ENVELOPE_SCHEMA,
    decode_frame,
    encode_frame,
)
from kumo_core.diskgraph.sampling_service import (
    SamplingService,
    SamplingServiceConfig,
)
from kumo_core.diskgraph.sampling_task import task_files, upload_task
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
            shutil.copytree(source, root, ignore=shutil.ignore_patterns("train_tables"))
            service = SamplingService(SamplingServiceConfig(root))
            services.append(service)
            cleanup.callback(service.close)
        runtime = RemoteRuntimeIndex(
            prepared_task, [service.endpoint() for service in services], 42, 1
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
        assert all(service.health()["exchange_streams"] == 1 for service in services)


def test_credits_bound_production_and_cancel_early(
    prepared_task: str, tmp_path: Path
) -> None:
    """A paused trainer bounds native work; cancellation frees its stream permit."""
    root = tmp_path / "replica"
    shutil.copytree(
        Path(prepared_task).parent / "graph",
        root,
        ignore=shutil.ignore_patterns("train_tables"),
    )
    service = SamplingService(SamplingServiceConfig(root))
    runtime = RemoteRuntimeIndex(prepared_task, service.endpoint(), 42, 1)
    try:
        iterator, count = runtime.iter(**{**_options(), "batch_size": 1})
        assert count == 6
        deadline = time.monotonic() + 5
        while service.health()["batches"] != 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)
        assert service.health()["batches"] == 2
        iterator.close()
        _wait_for_idle(service)
        complete, _ = runtime.iter(**_options())
        assert len(list(complete)) == 3
        _wait_for_idle(service)
    finally:
        runtime.close()
        service.close()


def test_task_transfer_is_reused_across_epochs(
    prepared_task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task files transfer once while each epoch gets a fresh independent stream."""
    root = tmp_path / "replica"
    shutil.copytree(
        Path(prepared_task).parent / "graph",
        root,
        ignore=shutil.ignore_patterns("train_tables"),
    )
    service = SamplingService(SamplingServiceConfig(root))
    runtime = RemoteRuntimeIndex(prepared_task, service.endpoint(), 42, 1)
    transfer = MagicMock(wraps=upload_task)
    monkeypatch.setattr("kumo_core.diskgraph.remote_sampling.upload_task", transfer)
    try:
        for epoch in range(3):
            iterator, _ = runtime.iter(**{**_options(), "epoch": epoch})
            assert len(list(iterator)) == 3
        assert transfer.call_count == 1
        assert service.health()["exchange_streams"] == 3
        local_task = Path(prepared_task).parent / "graph/train_tables/remote_test"
        assert task_files(root / "train_tables/remote_test") == task_files(local_task)
    finally:
        runtime.close()
        service.close()


@pytest.mark.parametrize("credits", [0, -1, 3, True])
def test_worker_rejects_invalid_credits(prepared_task: str, credits: int) -> None:
    """Malformed flow control fails before native batch production."""
    service = SamplingService(
        SamplingServiceConfig(Path(prepared_task).parent / "graph")
    )
    request = {
        "kind": "push_iter",
        "version": 4,
        "extension_ref": "train_tables/remote_test",
        "engine_seed": 42,
        "num_threads": 1,
        "options": {
            "stage_table": "train_seeds",
            "sampling_plan_json": '{"layers":[]}',
            "num_workers": 1,
            "worker_id": 0,
            "shuffle": False,
            "epoch": 0,
            "rnd_seed": 42,
            "batch_size": 1,
            "entity_node_type": "users",
        },
    }
    try:
        with flight.connect(service.endpoint()) as client:
            writer, reader = client.do_exchange(
                flight.FlightDescriptor.for_command(orjson.dumps(request))
            )
            writer.begin(ENVELOPE_SCHEMA)
            assert decode_frame(reader.read_chunk().data)[0]["num_batches"] == 6
            writer.write_batch(encode_frame({"kind": "credit", "batches": credits}, []))
            with pytest.raises(
                (pa.ArrowInvalid, flight.FlightServerError), match="credits"
            ):
                reader.read_chunk()
            reader.cancel()
        _wait_for_idle(service)
        assert service.health()["batches"] == 0
    finally:
        service.close()


@pytest.mark.parametrize("completion", [None, {"kind": "end", "batches": 0}])
def test_push_rejects_truncated_epochs(completion: dict[str, Any] | None) -> None:
    """Matching a received-batch count alone cannot certify a complete shard."""
    client = MagicMock()
    reader = MagicMock()
    reader.read_chunk.side_effect = [
        MagicMock(
            data=encode_frame(
                {"kind": "iterator", "version": 4, "num_samples": 4, "num_batches": 2},
                [],
            )
        ),
        StopIteration()
        if completion is None
        else MagicMock(data=encode_frame(completion, [])),
    ]
    client.do_exchange.return_value = (MagicMock(), reader)
    iterator = RemoteTaskIterator(
        "grpc://unused:1", {"kind": "push_iter", "version": 4}, client=client
    )
    with pytest.raises(ValueError, match="completion"):
        next(iterator)
    reader.cancel.assert_called_once()


def test_task_hash_is_canonical_and_rejects_symlinks(tmp_path: Path) -> None:
    """Nested file names hash consistently and symbolic links are never followed."""
    root = tmp_path / "task"
    (root / "a").mkdir(parents=True)
    (root / "a/data").write_bytes(b"content")
    (root / "a.bin").write_bytes(b"other")
    first = task_files(root)
    assert first[1] == [("a.bin", 5), ("a/data", 7)]
    assert task_files(root) == first
    (root / "linked").symlink_to(root / "a.bin")
    with pytest.raises(ValueError, match="symlinks"):
        task_files(root)


def test_upload_rejects_unacknowledged_checksum(tmp_path: Path) -> None:
    """The trainer requires confirmation before an uploaded task can be used."""
    root = tmp_path / "task"
    root.mkdir()
    (root / "data").write_bytes(b"content")
    client = MagicMock()
    client.do_action.return_value = [flight.Result(b"")]
    writer, reader = MagicMock(), MagicMock()
    reader.read.return_value = pa.py_buffer(b"wrong")
    client.do_put.return_value = writer, reader
    with pytest.raises(ValueError, match="acknowledged"):
        upload_task(client, root, "train_tables/task", task_files(root))


def test_replicas_produce_concurrently(
    prepared_task: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocked first feeder does not prevent another feeder producing its shard."""
    second_finished = threading.Event()
    with ExitStack() as cleanup:
        services = []
        for replica in range(2):
            service = SamplingService(
                SamplingServiceConfig(Path(prepared_task).parent / "graph")
            )
            cleanup.callback(service.close)
            services.append(service)
            original = service._open

            def open_shard(
                request: dict[str, Any],
                original: Any = original,
                replica: int = replica,
            ) -> MagicMock:
                native = original(request)
                wrapped = MagicMock(wraps=native)

                def sample() -> dict[str, Any]:
                    if replica == 0:
                        assert second_finished.wait(5), (
                            "Feeder production was serialized"
                        )
                    batch = next(native)
                    if replica == 1:
                        second_finished.set()
                    return batch

                wrapped.__next__.side_effect = sample
                return wrapped

            monkeypatch.setattr(service, "_open", open_shard)
        runtime = RemoteRuntimeIndex(
            prepared_task, [s.endpoint() for s in services], 42, 1
        )
        cleanup.callback(runtime.close)
        actual, _ = runtime.iter(**{**_options(), "batch_size": 1})
        expected, _ = diskgraph.RuntimeIndex(prepared_task, 42, 1).iter(
            **{**_options(), "batch_size": 1}
        )
        parity._assert_equal(list(expected), list(actual))
        assert second_finished.is_set()


def test_close_interrupts_a_blocked_push_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation wakes and joins a reader waiting for a remote batch."""
    from kumo_core.diskgraph import remote_sampling

    waiting, cancelled = threading.Event(), threading.Event()
    stream = MagicMock()
    stream.num_samples.return_value = 1

    def read() -> dict[str, Any]:
        waiting.set()
        assert cancelled.wait(5), "Push reader was not cancelled"
        raise flight.FlightCancelledError("cancelled")

    stream.__next__.side_effect = read
    stream.close.side_effect = cancelled.set
    monkeypatch.setattr(
        remote_sampling, "RemoteTaskIterator", MagicMock(return_value=stream)
    )
    owner = MagicMock()
    owner.endpoints = ("grpc://unused:1",)
    owner._epoch_only = set()
    owner._streams = set()
    iterator = remote_sampling.RemotePushIterator(
        owner, "train_tables/task", {"num_workers": 1, "worker_id": 0}
    )
    try:
        assert waiting.wait(5)
        pending = iterator._pending[0][1]
    finally:
        iterator.close()
    with pytest.raises(flight.FlightCancelledError):
        pending.result()
    assert not owner._streams


@pytest.mark.parametrize("failure", ["path", "truncated", "checksum", "completion"])
def test_invalid_task_transfer_is_not_published(
    prepared_task: str, failure: str
) -> None:
    """Bad paths, missing data and corrupt identities never create a usable task."""
    service = SamplingService(
        SamplingServiceConfig(Path(prepared_task).parent / "graph")
    )
    frames = [
        encode_frame({"kind": "file", "path": "data", "size": 3}, []),
        encode_frame({"kind": "data"}, [b"abc"]),
        encode_frame({"kind": "end"}, []),
    ]
    if failure == "path":
        frames[0] = encode_frame({"kind": "file", "path": "../escape", "size": 3}, [])
    elif failure == "truncated":
        frames[1] = encode_frame({"kind": "data"}, [b"a"])
    elif failure == "completion":
        frames.pop()
    reader = MagicMock()
    reader.__iter__.return_value = iter([MagicMock(data=frame) for frame in frames])
    context = MagicMock()
    context.is_cancelled.return_value = False
    descriptor = flight.FlightDescriptor.for_command(
        orjson.dumps(
            {
                "kind": "prepare_task",
                "version": 4,
                "extension_ref": "train_tables/invalid-task",
                "sha256": "bad",
            }
        )
    )
    try:
        with pytest.raises(ValueError):
            service.do_put(context, descriptor, reader, MagicMock())
        assert not (service._root / "train_tables/invalid-task").exists()
        assert "train_tables/invalid-task" not in service._tasks
    finally:
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
        remote = RemoteRuntimeIndex(
            prepared_task, [s.endpoint() for s in services], 42, 1
        )
        cleanup.callback(remote.close)
        options = {**_options(), "rhs_seed_node_type": "users", "rhs_seed_time": 0}
        expected, count = diskgraph.RuntimeIndex(prepared_task, 42, 1).iter(**options)
        actual, remote_count = remote.iter(**options)
        assert count == remote_count == 8
        parity._assert_equal(list(expected), list(actual))
        assert [s.health()["task_requests"] for s in services] == [1, 0, 0]
