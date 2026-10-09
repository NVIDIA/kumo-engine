# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""check_gpu_ready must fail loudly (not silently CPU-fall-back) when a GPU
is required but unavailable — the single-host silent-GPU-loss issue."""
import pytest
from kumo_core.automl.error import check_gpu_ready


def test_silent_skip_when_gpu_not_required(monkeypatch):
    # Default / require_gpu=False preserves the CPU-only behavior (no raise).
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    monkeypatch.setattr('torch.backends.mps.is_available', lambda: False)
    check_gpu_ready(require_gpu=False)


def test_raises_when_gpu_required_but_absent(monkeypatch):
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    monkeypatch.setattr('torch.backends.mps.is_available', lambda: False)
    with pytest.raises(RuntimeError, match='GPU required'):
        check_gpu_ready(require_gpu=True)


def test_env_var_enables_require(monkeypatch):
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    monkeypatch.setattr('torch.backends.mps.is_available', lambda: False)
    monkeypatch.setenv('KUMO_REQUIRE_GPU', 'true')
    with pytest.raises(RuntimeError, match='GPU required'):
        check_gpu_ready()


def test_mps_gpu_readiness(monkeypatch):
    devices = []
    monkeypatch.setattr('torch.mps.synchronize', lambda: None)
    monkeypatch.setattr('torch.mps.empty_cache', lambda: None)
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    monkeypatch.setattr('torch.backends.mps.is_available', lambda: True)
    monkeypatch.setattr(
        'torch.zeros', lambda *args, **kwargs: devices.append(kwargs['device']))

    check_gpu_ready(require_gpu=True)
    assert devices == ['mps']
