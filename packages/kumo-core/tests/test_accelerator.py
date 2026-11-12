# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core import accelerator as acc


def host(monkeypatch, system, release=''):
    monkeypatch.setattr(acc.platform, 'system', lambda: system)
    monkeypatch.setattr(acc.platform, 'mac_ver',
                        lambda: (release, ('', '', ''), ''))
    monkeypatch.setattr(acc, '_warned_auto_cpu', False)


@pytest.mark.parametrize('choice', ['mps', 'gpu'])
def test_mps_before_macos_15_raises_a_clear_error(monkeypatch, choice):
    host(monkeypatch, 'Darwin', '14.7.1')
    with pytest.raises(acc.UnsupportedAcceleratorError,
                       match=r'requires macOS 15 or later, but this host '
                       r'runs macOS 14\.7\.1'):
        acc.resolve_accelerator(choice)


def test_auto_before_macos_15_uses_the_cpu(monkeypatch, caplog):
    host(monkeypatch, 'Darwin', '14.7.1')
    assert acc.resolve_accelerator('auto') == 'cpu'
    assert acc.resolve_accelerator('auto') == 'cpu'
    assert acc.resolve_accelerator('cpu') == 'cpu'
    assert sum('requires macOS 15' in r.getMessage()
               for r in caplog.records) == 1


@pytest.mark.parametrize('system,release', [
    ('Darwin', '15.0'),
    ('Darwin', '26.1'),
    ('Darwin', ''),
    ('Linux', ''),
    ('Windows', ''),
])
@pytest.mark.parametrize('choice', ['auto', 'cpu', 'gpu', 'mps'])
def test_other_hosts_keep_the_choice(monkeypatch, system, release, choice):
    host(monkeypatch, system, release)
    assert acc.resolve_accelerator(choice) == choice


def test_required_gpu_rejects_mps_before_macos_15(monkeypatch):
    torch = pytest.importorskip('torch')
    from kumo_core.automl import error

    monkeypatch.setattr(acc, '_macos_release', lambda: '14.7.1')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    with pytest.raises(RuntimeError, match='MPS requires macOS 15'):
        error.check_gpu_ready(max_retries=1, delay_seconds=0,
                              require_gpu=True)
    # Without the requirement the check skips MPS instead of probing it.
    error.check_gpu_ready(max_retries=1, delay_seconds=0, require_gpu=False)
