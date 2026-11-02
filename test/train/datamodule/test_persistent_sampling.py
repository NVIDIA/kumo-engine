# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared runtime ownership and sampling stream lifetime."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest
from kumo_core.diskgraph import runtime_owner
from kumo_core.diskgraph.runtime_owner import RuntimeOwner


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    """Keep synthetic sampling tests independent of application services."""
    return False


@pytest.mark.parametrize("endpoints", [None, ["grpc://a:1"]])
def test_module_owns_stream_lifetime_across_stage_loaders(
    monkeypatch: pytest.MonkeyPatch,
    endpoints: list[str] | None,
) -> None:
    """Epoch iteration keeps shared streams open; job shutdown releases them."""
    from kumo_core.diskgraph.lightning_data import (
        DiskgraphLightningDataModule,
        DiskgraphTaskBatchIterable,
        _ShuffleStreamState,
    )

    source = MagicMock()
    reopened = MagicMock()
    source.iter.side_effect = lambda **_: (({"task": {}} for _ in range(1)), 1)
    factory = MagicMock(side_effect=[source, reopened])
    monkeypatch.setattr(runtime_owner, "RuntimeIndex", factory)
    module = object.__new__(DiskgraphLightningDataModule)
    module._runtime_owner = RuntimeOwner(
        "manifest.json", 42, sampling_endpoints=endpoints
    )
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
            runtime_owner=module._runtime_owner,
        )
        assert list(dataset) == [{"task": {}}]
        assert dataset._runtime_index() is source
        dataset.close()
    factory.assert_called_once()
    source.close.assert_not_called()
    module.shutdown_dataloader_workers("train")
    source.close.assert_not_called()
    module.shutdown_dataloader_workers()
    source.close.assert_called_once()
    assert dataset._runtime_index() is reopened
    assert factory.call_count == 2
    module.shutdown_dataloader_workers()
    reopened.close.assert_called_once()


def test_failed_worker_shutdown_still_closes_sampling_streams() -> None:
    """A DataLoader cleanup error must not strand persistent network streams."""
    from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule

    module = object.__new__(DiskgraphLightningDataModule)
    module._worker_pool_registry = MagicMock()
    module._worker_pool_registry.shutdown.side_effect = OSError("worker cleanup failed")
    source = MagicMock()
    module._runtime_owner = source
    with pytest.raises(OSError, match="cleanup failed"):
        module.shutdown_dataloader_workers()
    source.close.assert_called_once()


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
    owner = RuntimeOwner("manifest.json", 42)
    monkeypatch.setattr(runtime_owner, "RuntimeIndex", MagicMock(return_value=source))
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
        runtime_owner=owner,
        **options,
    )
    iterator = iter(dataset)
    assert isinstance(iterator, Generator)
    next(iterator)
    iterator.close()
    batch_iterator.close.assert_called_once()
    dataset.close()
    source.close.assert_not_called()
    owner.close()
    source.close.assert_called_once()
