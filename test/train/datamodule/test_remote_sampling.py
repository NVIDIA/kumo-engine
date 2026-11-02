# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared manifest normalization and native local/remote backend selection."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import diskgraph
import orjson
import pytest
from kumo_core.diskgraph import runtime_owner
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
    """Keep native-wrapper tests independent of application services."""
    return False


def test_remote_backend_reuses_shared_option_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A portable manifest builds a native request without opening local graph files."""
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
    local_factory = MagicMock()
    monkeypatch.setattr("diskgraph._c.RuntimeIndex", local_factory)
    backend = MagicMock()
    backend.iter.return_value.num_samples.return_value = 3
    backend_factory = MagicMock(return_value=backend)
    monkeypatch.setattr("diskgraph._c.RemoteRuntimeIndex", backend_factory)
    request_factory = MagicMock()
    monkeypatch.setattr("diskgraph._c.SamplingRequest", request_factory)
    index = runtime.RuntimeIndex(
        "portable.json",
        rnd_seed=42,
        num_threads=1,
        sampling_endpoints="grpc://sampler:50051",
    )
    iterator, count = index.iter(
        stage="train",
        sampling_plan={"layers": []},
        num_workers=2,
        worker_id=1,
        shuffle=True,
        epoch=7,
        batch_size=4,
        replace=True,
        weighted_seed_sampling={"num_samples": 3, "true_weight": 2},
    )
    assert count == 3
    assert iterator is backend.iter.return_value
    local_factory.assert_not_called()
    backend_factory.assert_called_once_with(["grpc://sampler:50051"])
    kwargs = request_factory.call_args.kwargs
    assert request_factory.call_args.args == ("train_tables/task",)
    assert kwargs["num_workers"] == 2 and kwargs["worker_id"] == 1
    assert kwargs["engine_seed"] == 42 and kwargs["epoch"] == 7
    assert kwargs["use_seed_timestamps"] is False
    weights = orjson.loads(kwargs["weighted_seed_sampling_json"])
    assert (
        weights["buckets"][0]["node_ids_path"]
        == "/trainer/graph/train_tables/task/train_seeds/false.bin"
    )
    backend.iter.assert_called_once_with(request_factory.return_value, "/trainer/graph")
    index.close()
    index.close()
    backend.close.assert_called_once()


@pytest.mark.parametrize("endpoints", [[], [""], ["grpc://a:1", "grpc://a:1"]])
def test_invalid_replicas_fail_before_manifest_io(endpoints: list[str]) -> None:
    """Validate endpoint identities before loading even a nonexistent manifest."""
    with pytest.raises(ValueError, match="endpoint"):
        diskgraph.RuntimeIndex("missing.json", sampling_endpoints=endpoints)


def test_closed_wrapper_cannot_reopen_native_resources() -> None:
    """Closed indexes fail before resolving manifests or starting connections."""
    index = diskgraph.RuntimeIndex(
        "missing.json", sampling_endpoints="grpc://127.0.0.1:1"
    )
    index.close()
    with pytest.raises(diskgraph.DiskgraphConfigError, match="closed"):
        index.iter(
            stage="train",
            sampling_plan={},
            num_workers=1,
            worker_id=0,
            shuffle=False,
            epoch=0,
            batch_size=1,
        )


@pytest.mark.parametrize("mode", ["uncapped", "capped", "stage_targets", "grouped"])
def test_iterable_cancels_remote_streams_on_early_exit(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Every scheduling branch owns and cancels its remote streams."""
    source = MagicMock()
    batch_iterator = MagicMock()
    batch_iterator.__iter__.side_effect = lambda: iter([{"task": {}}] * 3)
    batch_iterator.__next__.side_effect = [{"task": {}}] * 3
    source.iter.return_value = (batch_iterator, 3)
    source_factory = MagicMock(return_value=source)
    monkeypatch.setattr(runtime_owner, "RuntimeIndex", source_factory)
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
    source.close.assert_not_called()
    batch_iterator.close.assert_called()
    assert "remote_sampling_endpoint" not in source.iter.call_args.kwargs
    assert dataset._runtime_index() is source
    dataset.close()
    dataset.close()
    source.close.assert_called_once()


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
