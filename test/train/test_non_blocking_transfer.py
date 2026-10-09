# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The host-to-device copy must be asynchronous only when it is safe.

An async copy out of page-locked memory is sound because PyTorch's caching
host allocator will not recycle a pinned block until the copy that reads it
has completed. Neither condition holds for pageable memory, so the guard is
what makes the optimisation correct rather than merely fast -- which is why
it is worth pinning in a test.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch
from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule


class _RecordingBatch:
    """Stands in for ``HeteroData``: records how ``.to()`` was called."""
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def to(self, device: Any, **kwargs: Any) -> "_RecordingBatch":
        self.calls.append({"device": device, **kwargs})
        return self


class _NoKwargBatch:
    """A batch whose ``.to()`` predates ``non_blocking``."""
    def to(self, device: Any) -> "_NoKwargBatch":
        return self


class _ExplodingBatch:
    """A batch whose ``.to()`` raises a TypeError of its own."""
    def to(self, device: Any, **kwargs: Any) -> Any:
        raise TypeError("something genuinely broken inside to()")


def _transfer(pin_memory: bool, batch: Any, device: str) -> Any:
    # Built via __new__ so the test does not have to supply a manifest,
    # mapper bank or sampling plan -- the hook reads one attribute and
    # nothing else.
    dm = DiskgraphLightningDataModule.__new__(DiskgraphLightningDataModule)
    dm.pin_memory = pin_memory
    return DiskgraphLightningDataModule.transfer_batch_to_device(
        dm, batch, torch.device(device), 0)


def test_pinned_cuda_transfer_is_non_blocking() -> None:
    batch = _RecordingBatch()
    _transfer(True, batch, "cuda")
    assert batch.calls == [{
        "device": torch.device("cuda"),
        "non_blocking": True
    }]


def test_unpinned_loader_keeps_the_blocking_path() -> None:
    # Without pinned memory an async copy is silently synchronous anyway, and
    # the guard keeps this setting from drifting away from the loader's.
    batch = _RecordingBatch()
    _transfer(False, batch, "cuda")
    assert batch.calls == [] or "non_blocking" not in batch.calls[0]


def test_cpu_device_keeps_the_blocking_path() -> None:
    batch = _RecordingBatch()
    _transfer(True, batch, "cpu")
    assert batch.calls == [] or "non_blocking" not in batch.calls[0]


def test_batch_without_the_kwarg_falls_back() -> None:
    # Older batch types must still be transferred, not rejected.
    assert _transfer(True, _NoKwargBatch(), "cuda") is not None


def test_a_real_typeerror_is_not_swallowed() -> None:
    # The fallback exists for a signature mismatch only. A TypeError raised
    # INSIDE the transfer is a genuine bug, and routing it quietly down the
    # default path would hide it.
    with pytest.raises(TypeError, match="genuinely broken"):
        _transfer(True, _ExplodingBatch(), "cuda")
