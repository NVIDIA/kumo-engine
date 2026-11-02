# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-local runtime sharing and serialization regressions."""

from __future__ import annotations

import pickle
from unittest.mock import MagicMock

import diskgraph
import pytest
from kumo_core.diskgraph import runtime_owner
from kumo_core.diskgraph.lightning_data import (
    DiskgraphTaskBatchIterable,
    _ShuffleStreamState,
)
from kumo_core.diskgraph.runtime_owner import RuntimeOwner


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    """Keep ownership tests independent of application services."""
    return False


@pytest.mark.parametrize("endpoints", [None, "grpc://a:1", ["grpc://a:1"]])
def test_lazy_reuse_and_reopen(
    monkeypatch: pytest.MonkeyPatch, endpoints: str | list[str] | None
) -> None:
    """Both backends share a lazy handle until explicit owner shutdown."""
    first, second = MagicMock(), MagicMock()
    factory = MagicMock(side_effect=[first, second])
    monkeypatch.setattr(diskgraph, "RuntimeIndex", factory)
    owner = RuntimeOwner("manifest.json", 42, sampling_endpoints=endpoints)
    owner.close()
    factory.assert_not_called()
    assert owner.get() is first
    assert owner.get() is first
    expected: dict[str, object] = {"rnd_seed": 42, "num_threads": None}
    if endpoints is not None:
        expected["sampling_endpoints"] = endpoints
    factory.assert_called_once_with("manifest.json", **expected)
    owner.close()
    owner.close()
    first.close.assert_called_once()
    assert owner.get() is second
    owner.close()
    second.close.assert_called_once()


def test_pickle_drops_native_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pickling never serializes or closes a live native handle."""
    source, restored_source = MagicMock(), MagicMock()
    monkeypatch.setattr(
        diskgraph, "RuntimeIndex", MagicMock(side_effect=[source, restored_source])
    )
    owner = RuntimeOwner("manifest.json", 42)
    assert owner.get() is source
    restored = pickle.loads(pickle.dumps(owner))
    source.close.assert_not_called()
    assert owner.get() is source
    assert restored.get() is restored_source
    owner.close()
    restored.close()


@pytest.mark.parametrize("close_before_get", [False, True])
def test_pid_change_discards_inherited_handle(
    monkeypatch: pytest.MonkeyPatch, close_before_get: bool
) -> None:
    """A child process neither uses nor explicitly closes its parent's index."""
    parent, child = MagicMock(), MagicMock()
    monkeypatch.setattr(
        diskgraph, "RuntimeIndex", MagicMock(side_effect=[parent, child])
    )
    monkeypatch.setattr(runtime_owner.os, "getpid", lambda: 1)
    owner = RuntimeOwner("manifest.json", 42)
    assert owner.get() is parent
    monkeypatch.setattr(runtime_owner.os, "getpid", lambda: 2)
    if close_before_get:
        owner.close()
    assert owner.get() is child
    parent.close.assert_not_called()
    owner.close()
    child.close.assert_called_once()


def test_standalone_dataset_shares_index_across_epochs_and_pickle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Epochs close their iterators; standalone close owns the cached index."""
    source, restored_source = MagicMock(), MagicMock()
    source.iter.side_effect = lambda **_: (({"task": {}} for _ in range(1)), 1)
    factory = MagicMock(side_effect=[source, restored_source])
    monkeypatch.setattr(diskgraph, "RuntimeIndex", factory)
    dataset = DiskgraphTaskBatchIterable(
        manifest_path="manifest.json",
        stage="train",
        sampling_plan={},
        batch_size=1,
        shuffle=False,
        drop_last=False,
        base_seed=42,
        shuffle_state=_ShuffleStreamState(),
        num_seeds=1,
    )
    assert list(dataset) == [{"task": {}}]
    assert list(dataset) == [{"task": {}}]
    factory.assert_called_once()
    source.close.assert_not_called()
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._runtime_index() is restored_source
    restored.close()
    restored_source.close.assert_called_once()
    source.close.assert_not_called()
    dataset.close()
    dataset.close()
    source.close.assert_called_once()
