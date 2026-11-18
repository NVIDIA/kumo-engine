# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from kumo_core.training import config


@pytest.mark.parametrize('platform', ['win32', 'darwin'])
def test_default_workers_without_statvfs(monkeypatch, platform):
    for name in ('PYTEST_CURRENT_TEST', 'GITHUB_ACTIONS', 'SYNC_DATA_LOADER'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config.sys, 'platform', platform)
    monkeypatch.delattr(config.os, 'statvfs', raising=False)
    monkeypatch.setattr(torch.multiprocessing, 'cpu_count', lambda: 8)

    assert config.LoaderKwargs(batch_size=1024).num_workers == 8


@pytest.mark.parametrize('capacity, expected', [(0, 0), (1, 0), (3, 3), (8, 8)])
def test_linux_workers_respect_shared_memory(monkeypatch, capacity, expected):
    monkeypatch.setattr(config.sys, 'platform', 'linux')
    monkeypatch.setattr(
        config.os, 'statvfs',
        lambda path: SimpleNamespace(f_blocks=capacity, f_frsize=256 * 1024**2),
        raising=False)

    if expected < 8:
        with pytest.warns(RuntimeWarning, match='reducing DataLoader'):
            assert config.shm_safe_num_workers(8) == expected
    else:
        assert config.shm_safe_num_workers(8) == expected


def test_missing_linux_shared_memory_preserves_workers(monkeypatch):
    monkeypatch.setattr(config.sys, 'platform', 'linux')

    def unavailable(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(config.os, 'statvfs', unavailable, raising=False)
    assert config.shm_safe_num_workers(8) == 8
