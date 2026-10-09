# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``/dev/shm``-aware DataLoader worker sizing.

DataLoader workers pass batches through shared memory, so a container on
Docker's 64 MiB default kills them mid-epoch with "No space left on device".
Worker count is sized against ``/dev/shm`` to prevent that.

The probe is injected, so these need no GPU and no real ``/dev/shm``.
"""
from contextlib import nullcontext

import kumo_core.training.config as trainer_config
import pytest
from kumo_core.diskgraph.stores import diskgraph_auto_config
from kumo_core.training.config import LoaderKwargs, shm_safe_num_workers

MIB = 1024**2
GIB = 1024**3

DOCKER_DEFAULT_SHM = 64 * MIB

ENV_MARKERS_FORCING_ZERO_WORKERS = (
    'PYTEST_CURRENT_TEST',
    'GITHUB_ACTIONS',
    'SYNC_DATA_LOADER',
)


@pytest.fixture
def fake_shm(monkeypatch):
    """Inject a fake ``/dev/shm`` size in bytes (``None`` = unprobeable)."""
    def _set(total_bytes):
        monkeypatch.setattr(trainer_config, '_dev_shm_bytes',
                            lambda: total_bytes)

    return _set


class TestShmSafeNumWorkers:
    @pytest.mark.parametrize('shm,expected', [
        (1 * GIB, 4),
        (64 * GIB, 8),
    ])
    def test_reduces_to_what_shm_can_back(self, shm, expected, fake_shm):
        fake_shm(shm)
        assert shm_safe_num_workers(8) == expected

    @pytest.mark.parametrize('shm', [DOCKER_DEFAULT_SHM, 300 * MIB])
    def test_falls_back_to_sync_loading_below_two_workers(self, shm, fake_shm):
        fake_shm(shm)
        with pytest.warns(RuntimeWarning, match='ipc=host'):
            assert shm_safe_num_workers(8) == 0

    def test_unprobeable_shm_does_not_clamp(self, fake_shm):
        fake_shm(None)
        assert shm_safe_num_workers(8) == 8


class TestDiskgraphAutoConfig:
    """Worker count is reduced where it is produced.

    Callers overwrite ``LoaderKwargs.num_workers`` with this value, so a
    limit applied elsewhere never reaches the DataLoader.
    """
    @pytest.mark.parametrize('estimated_edges,mode', [
        (300_000, 'standalone'),
        (1_000, 'fused'),
    ])
    def test_small_shm_forces_sync_loading(self, estimated_edges, mode,
                                           fake_shm):
        fake_shm(DOCKER_DEFAULT_SHM)
        with pytest.warns(RuntimeWarning):
            cfg = diskgraph_auto_config(64, int(533.7e9), int(172e9),
                                        estimated_edges)
        assert cfg['mode'] == mode
        assert cfg['num_workers'] == 0

    @pytest.mark.parametrize('shm,expected_workers,expected_rayon', [
        (DOCKER_DEFAULT_SHM, 0, 2),
        (512 * MIB, 2, 8),
        (1 * GIB, 4, 8),
        (2 * GIB, 8, 4),
        (4 * GIB, 16, 2),
    ])
    def test_recomputes_rayon_after_shm_reduction(self, shm, expected_workers,
                                                  expected_rayon, fake_shm):
        fake_shm(shm)
        warning = (pytest.warns(RuntimeWarning)
                   if expected_workers < 16 else nullcontext())
        with warning:
            cfg = diskgraph_auto_config(64, int(267e9), 2 * 1024**4,
                                        estimated_edges=49_152)
        assert cfg['num_workers'] == expected_workers
        assert cfg['rayon_threads'] == expected_rayon


def test_loader_kwargs_resolution_reduces_to_shm(fake_shm, monkeypatch):
    import torch

    for var in ENV_MARKERS_FORCING_ZERO_WORKERS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(torch.multiprocessing, 'cpu_count', lambda: 16)
    fake_shm(1 * GIB)

    with pytest.warns(RuntimeWarning):
        kwargs = LoaderKwargs.cast({'num_workers': None})

    assert kwargs.num_workers == 4
