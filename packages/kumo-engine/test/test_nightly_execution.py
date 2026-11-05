# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admit actual CUDA capacity without requiring one GPU model."""

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_check.py'
SPEC = importlib.util.spec_from_file_location('nightly_execution', SOURCE)
nightly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nightly)


@pytest.mark.parametrize('available,gpus,memory,passes', [
    (True, 1, 24 * 1024**3, True),
    (False, 1, 24 * 1024**3, False),
    (True, 0, 24 * 1024**3, False),
    (True, 2, 24 * 1024**3, False),
    (True, 1, 0, False),
])
def test_runtime_uses_cuda_capacity_without_requiring_a_gpu_model(
        monkeypatch, available, gpus, memory, passes):
    import torch

    import kumo.engine as kumo

    prefix = nightly.sys.prefix
    monkeypatch.setattr(
        nightly, 'sys',
        SimpleNamespace(platform='linux', version_info=(3, 12), prefix=prefix))
    monkeypatch.setattr(nightly, 'platform',
                        SimpleNamespace(machine=lambda: 'x86_64'))
    monkeypatch.setattr(kumo, '__file__',
                        str(Path(prefix) / 'lib/kumo/engine/__init__.py'))
    metadata = SimpleNamespace(
        distributions=lambda: [], version=lambda name: kumo.__version__,
        distribution=lambda name: SimpleNamespace(
            read_text=lambda filename: json.dumps(
                dict(archive_info=dict(hashes=dict(sha256='a' * 64))))))
    monkeypatch.setattr(nightly, 'importlib',
                        SimpleNamespace(metadata=metadata))
    monkeypatch.setattr(
        nightly, 'subprocess',
        SimpleNamespace(check_output=lambda *args, **kwargs: '580.0\n'))
    properties = SimpleNamespace(
        name='NVIDIA A10G', total_memory=memory,
        uuid='GPU-12345678-1234-1234-1234-123456789abc', major=8, minor=6)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: gpus)
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda i: properties)
    if passes:
        runtime = nightly.verify_runtime(kumo.__version__, None)
        assert runtime['gpu'] == 'NVIDIA A10G'
        assert runtime['gpu_memory_bytes'] == 24 * 1024**3
        assert runtime['compute_capability'] == [8, 6]
    else:
        with pytest.raises(ValueError):
            nightly.verify_runtime(kumo.__version__, None)


@pytest.mark.parametrize('dimension', [
    'cpus', 'ram_bytes', 'scratch_bytes', 'gpus', 'gpu_memory_bytes',
    'time_seconds'
])
def test_worker_rejects_insufficient_capacity_before_building_snapshot(
        tmp_path, monkeypatch, dimension):
    resources = nightly.sibling('nightly_resources')
    wheel = tmp_path / 'candidate.whl'
    wheel.write_bytes(b'candidate wheel')
    output = tmp_path / 'evidence'
    output.mkdir()
    args = SimpleNamespace(scale='100M', case='binary-normal', mode='candidate',
                           input_root=tmp_path / 'inputs', output_dir=output,
                           scratch=tmp_path / 'scratch', wheel=wheel,
                           expected_version='1.2.0',
                           build_receipt=tmp_path / 'build.json',
                           time_budget_seconds=60)
    case = dict(scale='100M', task='binary', arm='normal', seed=0,
                graph='adtech')
    configuration = dict(
        scales={'100M': dict(inputs={}, input_manifest_sha256='a' * 64)},
        reference_identity={})
    minimum = dict(cpus=2, ram_bytes=1024, scratch_bytes=2048, gpus=1,
                   gpu_memory_bytes=512, time_seconds=60)
    capacity = dict(minimum)
    capacity[dimension] -= 1
    records = dict(requirements=dict(cases={'100M/binary-normal': minimum}),
                   reference=dict(cases=[]))
    observed = dict(errors=[], status='MEASURED', peaks={})
    monkeypatch.setattr(resources, 'join_cgroup', lambda scope: None)
    monkeypatch.setattr(resources, 'track_resources',
                        lambda *arguments: nullcontext(observed))
    monkeypatch.setattr(resources, 'available_capacity',
                        lambda *arguments: capacity)
    monkeypatch.setattr(nightly, 'load_inputs',
                        lambda *arguments: dict(cases=[case]))
    monkeypatch.setattr(
        nightly, 'verify_runtime', lambda *arguments, **keywords: dict(
            version='1.2.0', wheel_sha256=nightly.digest(wheel)))
    monkeypatch.setattr(nightly, 'verify_installed_files', lambda *arguments: 1)
    monkeypatch.setattr(nightly.importlib.metadata, 'distribution',
                        lambda name: SimpleNamespace())
    monkeypatch.setattr(nightly, 'check_install',
                        lambda *arguments, **keywords: {})
    monkeypatch.setattr(nightly, 'runner_identity', lambda: 'b' * 64)
    built = []
    monkeypatch.setattr(nightly, 'build_graph',
                        lambda *arguments, **keywords: built.append(True))
    initial = dict(case, status='RUNNING', mode='candidate',
                   execution_id='one-case', bindings={})
    row = nightly._worker(args, configuration, records, initial,
                          tmp_path / 'dedicated-cgroup')
    assert built == []
    assert row['status'] == 'FAIL'
    assert row['error'].startswith('ValueError: Insufficient ' + dimension)
    saved = nightly.read(output / 'result.json')
    assert saved['status'] == 'FAIL'
    assert saved['error'] == row['error']
    assert saved['resources']['errors'] == []
