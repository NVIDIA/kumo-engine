# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared runtime ownership and sampling envelope validation."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
from kumo_core.diskgraph import remote_sampling as remote
from kumo_core.diskgraph.sampling_protocol import decode_frame, encode_frame


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    """Keep synthetic sampling tests independent of application services."""
    return False


def test_module_owns_stream_lifetime_across_stage_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Epoch iteration keeps shared streams open; job shutdown releases them."""
    from kumo_core.diskgraph.lightning_data import (
        DiskgraphLightningDataModule,
        DiskgraphTaskBatchIterable,
        _ShuffleStreamState,
    )

    source = MagicMock()
    factory = MagicMock(return_value=source)
    monkeypatch.setattr(remote, "RemoteRuntimeIndex", factory)
    module = object.__new__(DiskgraphLightningDataModule)
    module.runtime_options = {"remote_sampling_endpoint": ["grpc://a:1"]}
    module.base_seed = 42
    module._worker_pool_registry = MagicMock()
    for stage in ("train", "val"):
        dataset = DiskgraphTaskBatchIterable(
            manifest_path="manifest.json",
            stage=stage,
            sampling_plan={},
            batch_size=1,
            shuffle=False,
            drop_last=False,
            base_seed=42,
            shuffle_state=_ShuffleStreamState(),
            num_seeds=1,
            runtime_options=module.runtime_options,
        )
        module._bind_remote_runtime(dataset)
        monkeypatch.setattr(dataset, "_iter_batches", lambda: iter([{"task": {}}]))
        assert list(dataset) == [{"task": {}}]
        assert dataset._runtime_index() is source
        assert "_runtime_index_managed" not in dataset.__getstate__()
    factory.assert_called_once()
    source.close.assert_not_called()
    module.shutdown_dataloader_workers("train")
    source.close.assert_not_called()
    module.shutdown_dataloader_workers()
    source.close.assert_called_once()


def test_failed_worker_shutdown_still_closes_sampling_streams() -> None:
    """A DataLoader cleanup error must not strand persistent network streams."""
    from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule

    module = object.__new__(DiskgraphLightningDataModule)
    module._worker_pool_registry = MagicMock()
    module._worker_pool_registry.shutdown.side_effect = OSError("worker cleanup failed")
    source = MagicMock()
    module.__dict__["_remote_runtime_index"] = source
    with pytest.raises(OSError, match="cleanup failed"):
        module.shutdown_dataloader_workers()
    source.close.assert_called_once()


@pytest.mark.parametrize("frame", [None, pa.record_batch({"wrong": [1]})])
def test_frame_decoder_rejects_invalid_envelopes(frame: pa.RecordBatch | None) -> None:
    """Reject missing and schema-mismatched Flight payloads."""
    with pytest.raises(ValueError, match="envelope"):
        decode_frame(frame)


def test_frame_decoder_enforces_the_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce the inbound budget before parsing JSON or Arrow components."""
    from kumo_core.diskgraph import sampling_protocol

    frame = encode_frame({"kind": "sample"}, [b"data"])
    monkeypatch.setattr(sampling_protocol, "MAX_BATCH_BYTES", 1)
    with pytest.raises(ValueError, match="64 MiB"):
        decode_frame(frame)


@pytest.mark.parametrize("mode", ["uncapped", "capped", "stage_targets", "grouped"])
def test_epoch_cleanup_discards_prefetch_without_closing_shared_streams(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """Closing one loader's iterator must leave another stage's stream usable."""
    from kumo_core.diskgraph.lightning_data import (
        DiskgraphTaskBatchIterable,
        _ShuffleStreamState,
    )

    source = MagicMock()
    batch_iterator = MagicMock()
    batch_iterator.__iter__.return_value = iter([{"task": {}}] * 3)
    batch_iterator.__next__.side_effect = [{"task": {}}] * 3
    source.iter.return_value = (batch_iterator, 3)
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
        manifest_path="manifest.json",
        stage="train",
        sampling_plan={},
        batch_size=1,
        shuffle=False,
        drop_last=False,
        base_seed=42,
        shuffle_state=_ShuffleStreamState(),
        num_seeds=3,
        runtime_options={"remote_sampling_endpoint": "grpc://sampler:1"},
        **options,
    )
    monkeypatch.setattr(dataset, "_runtime_index", lambda: source)
    dataset.__dict__["_runtime_index_handle"] = source
    dataset.__dict__["_runtime_index_managed"] = True
    iterator = iter(dataset)
    assert isinstance(iterator, Generator)
    next(iterator)
    iterator.close()
    batch_iterator.close.assert_called_once()
    source.close.assert_not_called()
