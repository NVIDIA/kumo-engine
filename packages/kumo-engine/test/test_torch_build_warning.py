# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest
import torch
from kumo_core.utils import torch_build


@pytest.fixture(autouse=True)
def _uncached(monkeypatch):
    # The check runs once per process; every test needs a fresh call.
    torch_build.check_torch_build.cache_clear()
    monkeypatch.delenv('KUMO_CHECK_TORCH_BUILD', raising=False)
    yield
    torch_build.check_torch_build.cache_clear()


def _run(monkeypatch, version, *, cuda, driver):
    monkeypatch.setattr(torch, '__version__', version)
    monkeypatch.setattr(torch.version, 'cuda', cuda)
    monkeypatch.setattr(torch_build, 'nvidia_driver_present', lambda: driver)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        torch_build.check_torch_build()
    return [str(w.message) for w in caught]


@pytest.mark.parametrize('version,label', [
    ('2.15.0', None),
    ('2.7.0+cu126', 'cu126'),
    ('2.15.0+cpu', 'cpu'),
    ('2.15.0+xpu', 'xpu'),
    ('2.9.0+rocm6.2', 'rocm6.2'),
])
def test_local_label(version, label) -> None:
    assert torch_build.local_label(version) == label


@pytest.mark.parametrize('version,backend', [
    ('2.15.0', None),
    ('2.7.0+cu126', None),
    ('2.8.0+cu128', None),
    ('2.15.0+cpu', None),
    ('2.15.0+xpu', 'xpu'),
    ('2.9.0+rocm6.2', 'rocm'),
    ('2.9.0+hpu', 'hpu'),
])
def test_unsupported_backend(version, backend) -> None:
    assert torch_build.unsupported_backend(version) == backend


@pytest.mark.parametrize('version,backend', [
    ('2.15.0+xpu', 'Intel XPU'),
    ('2.9.0+rocm6.2', 'AMD ROCm'),
])
def test_unsupported_build_warns(monkeypatch, version, backend) -> None:
    caught = _run(monkeypatch, version, cuda=None, driver=False)
    assert len(caught) == 1
    assert backend in caught[0]
    assert version in caught[0]
    assert 'download.pytorch.org/whl/cu126' in caught[0]
    assert 'KUMO_CHECK_TORCH_BUILD=0' in caught[0]


@pytest.mark.parametrize('version,cuda', [
    ('2.15.0', '12.8'),
    ('2.7.0+cu126', '12.6'),
])
def test_cuda_build_is_silent(monkeypatch, version, cuda) -> None:
    assert _run(monkeypatch, version, cuda=cuda, driver=True) == []


def test_cpu_build_without_driver_is_silent(monkeypatch) -> None:
    # A documented, supported configuration for small jobs.
    assert _run(monkeypatch, '2.15.0+cpu', cuda=None, driver=False) == []


def test_apple_silicon_is_silent(monkeypatch) -> None:
    # macOS reports no CUDA and no NVIDIA driver, and uses MPS.
    assert _run(monkeypatch, '2.15.0', cuda=None, driver=False) == []


def test_missing_cuda_with_driver_warns(monkeypatch) -> None:
    caught = _run(monkeypatch, '2.15.0+cpu', cuda=None, driver=True)
    assert len(caught) == 1
    assert 'no CUDA support' in caught[0]
    assert 'NVIDIA driver' in caught[0]


def test_unsupported_build_reports_backend_not_cuda(monkeypatch) -> None:
    # The backend problem is the actionable one, so it wins over the
    # missing-CUDA message when both apply.
    caught = _run(monkeypatch, '2.15.0+xpu', cuda=None, driver=True)
    assert len(caught) == 1
    assert 'Intel XPU' in caught[0]


def test_env_var_silences(monkeypatch) -> None:
    monkeypatch.setenv('KUMO_CHECK_TORCH_BUILD', '0')
    assert _run(monkeypatch, '2.15.0+xpu', cuda=None, driver=True) == []


def test_warns_once_per_process(monkeypatch) -> None:
    monkeypatch.setattr(torch, '__version__', '2.15.0+xpu')
    monkeypatch.setattr(torch_build, 'nvidia_driver_present', lambda: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        torch_build.check_torch_build()
        torch_build.check_torch_build()
    assert len(caught) == 1


def test_never_raises(monkeypatch) -> None:
    def boom() -> bool:
        raise RuntimeError('probe failed')

    monkeypatch.setattr(torch, '__version__', '2.15.0+cpu')
    monkeypatch.setattr(torch.version, 'cuda', None)
    monkeypatch.setattr(torch_build, 'nvidia_driver_present', boom)
    torch_build.check_torch_build()


def test_driver_probe_handles_missing_paths(monkeypatch) -> None:
    monkeypatch.setattr(torch_build.os.path, 'exists', lambda path: False)
    assert torch_build.nvidia_driver_present() is False
