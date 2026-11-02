# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for complete remote task batches and iterator lifetime."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import orjson
import pyarrow as pa
import pytest
from kumo_core.diskgraph import remote_sampling as remote
from kumo_core.diskgraph.lightning_data import (
    DiskgraphTaskBatchIterable,
    _ShuffleStreamState,
)
from kumo_core.train.diskgraph_nested_trainer import (
    _runtime_options_from_sampler,
)
from kumo_core.training.config import SamplerKwargs


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    """Keep these tests independent of application services."""
    return False


def _request() -> remote.SamplingRequest:
    return remote.SamplingRequest(
        "train_tables/task",
        stage_table="train_seeds",
        sampling_plan_json='{"layers":[]}',
        batch_size=2,
    )


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    batches: list[dict[str, Any] | Exception],
    *,
    num_samples: int = 3,
) -> tuple[MagicMock, MagicMock]:
    reader = MagicMock()
    reader.num_samples = num_samples
    reader.__next__.side_effect = batches
    client = MagicMock()
    client.open.return_value = reader
    monkeypatch.setattr(remote, "SamplingClient", MagicMock(return_value=client))
    return client, reader


def test_complete_batches_preserve_arrow_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward native Arrow objects without Python serialization."""
    targets = pa.record_batch(
        [pa.array([[1, 2], None], pa.list_(pa.int64()))], names=["target"]
    )
    features = pa.record_batch(
        [pa.array(["a", "b"]).dictionary_encode()], names=["category"]
    )
    nodes = pa.array([3, 9], pa.uint64())
    value = {
        "task": {"target": targets},
        "graphs": {"main": {"node_ids": {"u": nodes}}},
        "raw_features": {"u": {"record_batches": [features]}},
    }
    client, reader = _install_transport(monkeypatch, [value, StopIteration()])
    request = _request()
    iterator = remote.RemoteTaskIterator("grpc://sampler:50051", request)
    client.open.assert_called_once_with(request)
    batches = list(iterator)
    assert iterator.num_samples() == 3
    assert batches[0] is value
    assert batches[0]["task"]["target"] is targets
    assert batches[0]["raw_features"]["u"]["record_batches"][0] is features
    assert batches[0]["graphs"]["main"]["node_ids"]["u"] is nodes
    reader.close.assert_called_once()
    client.close.assert_called_once()
    iterator.close()
    client.close.assert_called_once()


@pytest.mark.parametrize("num_batches", [0, 1])
def test_observes_final_rpc_error(
    monkeypatch: pytest.MonkeyPatch, num_batches: int
) -> None:
    """Receiving the advertised batch count cannot hide a final Flight failure."""
    batches = [{"task": {}, "graphs": {}, "raw_features": {}}] * num_batches
    client, reader = _install_transport(
        monkeypatch, [*batches, OSError("connection lost")]
    )
    iterator = remote.RemoteTaskIterator("grpc://sampler:50051", _request())
    with pytest.raises(OSError, match="connection lost"):
        list(iterator)
    assert reader.__next__.call_count == num_batches + 1
    client.close.assert_called_once()


def test_empty_shard_completes_at_flight_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    """A schema and successful EOF suffice for an empty shard."""
    client, reader = _install_transport(monkeypatch, [StopIteration()], num_samples=0)
    iterator = remote.RemoteTaskIterator("grpc://sampler:50051", _request())
    assert iterator.num_samples() == 0
    assert list(iterator) == []
    reader.__next__.assert_called_once()
    client.close.assert_called_once()


def test_normalizes_options_for_remote_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weighted sidecars become replica-relative and epoch options reach the worker."""
    import diskgraph.runtime_batch_iterator as runtime

    manifest = {
        "kind": "diskgraph_task_dataset",
        "graph": {"index_path": "/trainer/graph"},
        "task": {"is_temporal": False},
        "stages": {
            "train": {
                "table_name": "train_seeds",
                "index_path": "/trainer/graph/train_tables/task",
                "row_count": 3,
                "binary_label_index": {
                    "false_node_ids": "train_seeds/false.bin",
                    "true_node_ids": "train_seeds/true.bin",
                    "false_count": 2,
                    "true_count": 1,
                },
            }
        },
    }
    monkeypatch.setattr(runtime, "_load_json", MagicMock(return_value=manifest))
    native_iterator = MagicMock()
    native_iterator.num_samples.return_value = 3
    native_iterator.pseudo_filter_info.return_value = None
    native = MagicMock()
    native.iter.return_value = native_iterator
    factory = MagicMock(return_value=native)
    monkeypatch.setattr("diskgraph._c.RuntimeIndex", factory)
    stream = MagicMock()
    stream.num_samples.return_value = 3
    stream_factory = MagicMock(return_value=stream)
    monkeypatch.setattr(remote, "RemoteTaskIterator", stream_factory)
    request_factory = MagicMock()
    monkeypatch.setattr(remote, "SamplingRequest", request_factory)
    client, _ = _install_transport(monkeypatch, [])
    index = remote.RemoteRuntimeIndex("portable.json", "grpc://sampler:50051", 42, 1)
    iterator, count = index.iter(
        stage="train",
        sampling_plan={"layers": []},
        num_workers=2,
        worker_id=1,
        shuffle=True,
        epoch=7,
        batch_size=4,
        replace=True,
        selected_stage_node_ids=None,
        weighted_seed_sampling={"num_samples": 3, "true_weight": 2},
    )
    assert count == 3
    factory.assert_not_called()
    assert stream_factory.call_args.args[1] is request_factory.return_value
    assert request_factory.call_args.args == ("train_tables/task",)
    options = request_factory.call_args.kwargs
    assert options["engine_seed"] == 42
    assert options["num_threads"] == 1
    assert options["worker_id"] == 1
    assert options["epoch"] == 7
    assert options["replace"] is True
    assert options["use_seed_timestamps"] is False
    weights = orjson.loads(options["weighted_seed_sampling_json"])
    assert weights["buckets"][0]["node_ids_path"] == "train_seeds/false.bin"
    assert weights["buckets"][1]["mass"] == 2
    client.open.assert_not_called()
    index.close()
    assert iterator not in index._streams


@pytest.mark.parametrize("mode", ["uncapped", "capped", "stage_targets", "grouped"])
def test_iterable_cancels_remote_streams_on_early_exit(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Every scheduling branch owns and cancels its remote streams."""
    source = MagicMock()
    source.iter.side_effect = lambda **_: (iter([{"task": {}}] * 3), 3)
    source_factory = MagicMock(return_value=source)
    monkeypatch.setattr(remote, "RemoteRuntimeIndex", source_factory)
    options: dict[str, Any] = {}
    if mode == "capped":
        options["max_batches"] = 1
    elif mode == "stage_targets":
        options["stage_batch_targets"] = {"train": 1}
    elif mode == "grouped":
        options["group_by_timestamp"] = True
        monkeypatch.setattr(
            "kumo_core.diskgraph.lightning_data._stage_bucket_counts",
            lambda *_: [(1, 3)],
        )
    dataset = DiskgraphTaskBatchIterable(
        manifest_path="portable.json",
        stage="train",
        sampling_plan={"layers": []},
        batch_size=1,
        shuffle=False,
        drop_last=False,
        base_seed=42,
        shuffle_state=_ShuffleStreamState(),
        num_seeds=3,
        runtime_options={"remote_sampling_endpoint": "grpc://sampler:50051"},
        **options,
    )
    iterator = iter(dataset)
    assert isinstance(iterator, Generator)
    assert next(iterator) == {"task": {}}
    iterator.close()
    source.close.assert_called_once()
    assert "remote_sampling_endpoint" not in source.iter.call_args.kwargs
    assert "_runtime_index_handle" not in dataset.__getstate__()


def test_sampler_knob_defaults_to_local() -> None:
    """The training configuration forwards the endpoint only when enabled."""
    local = SamplerKwargs()
    assert local.remote_sampling_endpoint is None
    assert "remote_sampling_endpoint" not in _runtime_options_from_sampler(local)
    assert local.remote_sampling_num_threads == 1
    with pytest.raises(ValueError):
        SamplerKwargs(remote_sampling_num_threads=0)
    configured = SamplerKwargs(remote_sampling_endpoint="grpc://sampler:50051")
    assert _runtime_options_from_sampler(configured)["remote_sampling_endpoint"] == (
        "grpc://sampler:50051"
    )


def test_request_failure_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed RPC must release the connection even before the first frame."""
    client, _ = _install_transport(monkeypatch, [])
    client.open.side_effect = OSError("unreachable")
    with pytest.raises(OSError, match="unreachable"):
        remote.RemoteTaskIterator("grpc://sampler:50051", _request())
    client.close.assert_called_once()


def test_midstream_failure_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport failure propagates without reconnecting or replaying rows."""
    client, reader = _install_transport(monkeypatch, [OSError("broken")])
    iterator = remote.RemoteTaskIterator("grpc://sampler:50051", _request())
    with pytest.raises(OSError, match="broken"):
        next(iterator)
    client.open.assert_called_once()
    client.close.assert_called_once()
    reader.close.assert_called_once()


def test_source_closes_all_streams_when_cancellation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cleanup error on one stream must not leak the remaining streams."""
    source = remote.RemoteRuntimeIndex("portable.json", "grpc://sampler:50051")
    first_client, first_reader = _install_transport(monkeypatch, [])
    first = remote.RemoteTaskIterator(source.endpoint, _request())
    first_reader.close.side_effect = OSError("cancel failed")
    second_client, _ = _install_transport(monkeypatch, [])
    second = remote.RemoteTaskIterator(source.endpoint, _request())
    source._streams.update([first, second])
    first._owner = source
    second._owner = source

    with pytest.raises(ExceptionGroup, match="Failed to close") as error:
        source.close()

    assert len(error.value.exceptions) == 1
    assert not source._streams
    first_client.close.assert_called_once()
    second_client.close.assert_called_once()


@pytest.mark.parametrize("endpoints", [[], [""], ["grpc://a:1", "grpc://a:1"]])
def test_replica_endpoints_are_validated(
    monkeypatch: pytest.MonkeyPatch, endpoints: list[str]
) -> None:
    """Reject empty and duplicated pools before opening Flight connections."""
    monkeypatch.setattr(remote.RuntimeIndex, "__init__", MagicMock(return_value=None))
    connection = MagicMock()
    monkeypatch.setattr(remote, "SamplingClient", connection)
    with pytest.raises(ValueError, match="endpoint"):
        remote.RemoteRuntimeIndex("manifest.json", endpoints)
    connection.assert_not_called()
