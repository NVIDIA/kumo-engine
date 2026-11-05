# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for compact resource admission and isolated case accounting."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_resources.py'


def load_resources():
    assert SOURCE.is_file(), 'The shared resource module is missing'
    spec = importlib.util.spec_from_file_location('nightly_resources', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measured():
    return dict(
        format='kumo-engine-nightly-resources-v1', status='MEASURED',
        peaks=dict(cpu_cores=2.1, ram_bytes=100, scratch_bytes=200,
                   gpu_memory_bytes=50), seconds=9.1, errors=[],
        gpu_class='observed GPU', gpus=1, wheel_sha256='a' * 64,
        input_manifest_sha256='b' * 64, harness_sha256='c' * 64,
        memory_accounting=dict(method='cgroup-v2-memory.peak', dedicated=True,
                               path='/sys/fs/cgroup/kumo-nightly-' + 'd' * 32))


def test_compact_minima_are_independent_of_archived_gpu_name():
    resources = load_resources()
    proposal = resources.resource_requirements(measured())
    minimum = dict(cpus=3, ram_bytes=125, scratch_bytes=250, gpus=1,
                   gpu_memory_bytes=63, time_seconds=12)
    assert proposal['minimum'] == minimum
    assert proposal['measurement']['gpu_class'] == 'observed GPU'
    assert resources.admit_resources(minimum,
                                     dict(minimum,
                                          gpu_class='another GPU')) == minimum
    for dimension in minimum:
        with pytest.raises(ValueError, match=dimension):
            resources.admit_resources(
                minimum, dict(minimum, **{dimension: minimum[dimension] - 1}))


def test_gpu_memory_observes_only_the_selected_visible_device(monkeypatch):
    resources = load_resources()
    selected = 'GPU-22222222-2222-2222-2222-222222222222'
    monkeypatch.setenv('ONE_GPU_UUID', selected)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', selected)
    cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_properties=lambda device: SimpleNamespace(uuid=selected),
        mem_get_info=lambda device=None: (12 * 1024**3, 96 * 1024**3))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    assert resources._gpu_memory_info() == (12 * 1024**3, 96 * 1024**3)
    monkeypatch.setenv('ONE_GPU_UUID',
                       'GPU-11111111-1111-1111-1111-111111111111')
    with pytest.raises(ValueError, match='selected GPU'):
        resources._gpu_memory_info()


def test_gpu_memory_rejects_multiple_visible_cuda_devices(monkeypatch):
    resources = load_resources()
    monkeypatch.delenv('ONE_GPU_UUID', raising=False)
    cuda = SimpleNamespace(device_count=lambda: 2,
                           mem_get_info=lambda device=None: (80, 100))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    with pytest.raises(ValueError, match='one visible CUDA GPU'):
        resources._gpu_memory_info()


@pytest.mark.parametrize('damage', [
    dict(status='INCOMPLETE'),
    dict(errors=['counter denied']),
    dict(memory_accounting={}),
    dict(seconds=float('nan')),
    dict(wheel_sha256='unbound'),
])
def test_incomplete_or_old_rss_measurements_cannot_make_requirements(damage):
    resources = load_resources()
    with pytest.raises(ValueError):
        resources.resource_requirements(dict(measured(), **damage))


@pytest.mark.parametrize('value', [
    None, [],
    dict(measured(), peaks=None),
    dict(measured(), memory_accounting=None),
    dict(measured(), harness_sha256=None)
])
def test_malformed_measurements_fail_as_invalid_proposals(value):
    resources = load_resources()
    with pytest.raises(ValueError):
        resources.resource_requirements(value)


def cgroup_fixture(resources, tmp_path, monkeypatch, *, peak=0):
    root = tmp_path / 'cgroup'
    root.mkdir()
    (root / 'cgroup.subtree_control').write_text('cpu memory\n')
    monkeypatch.setattr(resources, 'CGROUP_ROOT', root)
    monkeypatch.setattr(resources, '_current_cgroup', lambda: root)

    def create(parent):
        scope = parent / ('kumo-nightly-' + 'e' * 32)
        scope.mkdir()
        for name, text in {
                'memory.peak': str(peak),
                'cgroup.procs': '',
                'cgroup.events': 'populated 0\nfrozen 0\n',
                'cpu.stat': 'usage_usec 0\n',
        }.items():
            (scope / name).write_text(text)
        return scope

    monkeypatch.setattr(resources, '_create_scope', create)
    monkeypatch.setattr(resources, '_remove_scope', shutil.rmtree)
    return root


def test_dedicated_group_keeps_parent_outside_and_cleans_its_scope(
        tmp_path, monkeypatch):
    resources = load_resources()
    root = cgroup_fixture(resources, tmp_path, monkeypatch)
    with resources.dedicated_cgroup(root) as scope:
        assert scope.parent == root
        assert (scope / 'cgroup.procs').read_text() == ''
        assert resources.cgroup_memory_peak(scope) == 0
        assert resources._current_cgroup() == root
    assert not scope.exists()
    assert root.exists()


def test_reused_or_undelegated_scope_is_rejected(tmp_path, monkeypatch):
    resources = load_resources()
    root = cgroup_fixture(resources, tmp_path, monkeypatch, peak=17)
    with pytest.raises(ValueError, match='fresh'):
        with resources.dedicated_cgroup(root):
            pytest.fail('An old memory peak was admitted')
    (root / 'cgroup.subtree_control').write_text('cpu\n')
    with pytest.raises(ValueError, match='memory'):
        with resources.dedicated_cgroup(root):
            pytest.fail('Missing memory delegation was admitted')


def test_join_verifies_membership_and_is_idempotent(tmp_path, monkeypatch):
    resources = load_resources()
    root = cgroup_fixture(resources, tmp_path, monkeypatch)
    with resources.dedicated_cgroup(root) as scope:
        monkeypatch.setattr(
            resources, '_current_cgroup', lambda: scope
            if (scope / 'cgroup.procs').read_text() else root)
        resources.join_cgroup(scope)
        resources.join_cgroup(scope)
        assert (scope / 'cgroup.procs').read_text() == str(os.getpid())
        (scope / 'cgroup.procs').write_text('')


def test_join_cannot_fake_actual_kernel_membership(tmp_path, monkeypatch):
    resources = load_resources()
    root = cgroup_fixture(resources, tmp_path, monkeypatch)
    with resources.dedicated_cgroup(root) as scope:
        with pytest.raises(ValueError, match='membership'):
            resources.join_cgroup(scope)
        (scope / 'cgroup.procs').write_text('')


def test_scope_counters_cannot_follow_symlinks(tmp_path, monkeypatch):
    resources = load_resources()
    root = cgroup_fixture(resources, tmp_path, monkeypatch)
    outside = tmp_path / 'not-a-counter'
    outside.write_text('999')
    with resources.dedicated_cgroup(root) as scope:
        (scope / 'memory.peak').unlink()
        (scope / 'memory.peak').symlink_to(outside)
        with pytest.raises(ValueError, match='counter'):
            resources.cgroup_memory_peak(scope)
    assert outside.read_text() == '999'


def test_capacity_respects_ancestor_limits(tmp_path, monkeypatch):
    import psutil

    resources = load_resources()
    root = cgroup_fixture(resources, tmp_path, monkeypatch)
    child = root / 'case-parent'
    child.mkdir()
    (root / 'memory.max').write_text('1000')
    (root / 'memory.current').write_text('300')
    (root / 'cpu.max').write_text('150000 100000')
    (child / 'memory.max').write_text('max')
    (child / 'cpu.max').write_text('max 100000')
    monkeypatch.setattr(resources, '_current_cgroup', lambda: child)
    monkeypatch.setattr(resources.os, 'sched_getaffinity',
                        lambda pid: set(range(8)))
    monkeypatch.setattr(psutil, 'virtual_memory',
                        lambda: SimpleNamespace(available=5000))
    monkeypatch.setattr(shutil, 'disk_usage',
                        lambda path: SimpleNamespace(free=3000))
    monkeypatch.setattr(resources, '_gpu_memory_info', lambda: (400, 800))
    assert resources.available_capacity(tmp_path, dict(gpu_count=1),
                                        20) == dict(cpus=1.5, ram_bytes=700,
                                                    scratch_bytes=3000,
                                                    gpu_memory_bytes=400,
                                                    gpus=1, time_seconds=20)


def test_observer_uses_group_peak_and_retains_counter_errors(
        tmp_path, monkeypatch):
    resources = load_resources()
    root = cgroup_fixture(resources, tmp_path, monkeypatch)
    output, scratch = tmp_path / 'output', tmp_path / 'scratch'
    output.mkdir()
    scratch.mkdir()
    (scratch / 'data').write_bytes(b'x' * 4096)
    runtime = dict(gpu='pending', gpu_count=1)
    monkeypatch.setattr(resources, '_gpu_memory_info', lambda: (80, 100))
    with resources.dedicated_cgroup(root) as scope:
        monkeypatch.setattr(resources, '_current_cgroup', lambda: scope)
        with resources.track_resources(output, scratch, runtime,
                                       scope) as result:
            runtime['gpu'] = 'observed GPU'
            (scope / 'memory.peak').write_text('321')
            (scope / 'cpu.stat').write_text('usage_usec 100000\n')
        assert result['status'] == 'MEASURED'
        assert result['peaks']['ram_bytes'] == 321
        assert result['peaks']['gpu_memory_bytes'] == 20
        assert result['cpu_seconds'] == .1
        assert result['memory_accounting']['method'] == 'cgroup-v2-memory.peak'
        assert result['gpu_class'] == 'observed GPU'
        (scope / 'memory.peak').write_text('unreadable counter')
        with resources.track_resources(output, scratch, runtime, scope) as bad:
            pass
        assert bad['status'] == 'INCOMPLETE'
        assert bad['errors']


def test_external_deadline_kills_child_and_its_descendant(
        tmp_path, monkeypatch):
    import psutil

    resources = load_resources()
    root = cgroup_fixture(resources, tmp_path, monkeypatch)
    # Replace only the kernel join boundary; execute the real timeout supervisor.
    monkeypatch.setattr(resources, '_joined_command',
                        lambda command, scope: command)
    pid_file = tmp_path / 'child.pid'
    script = (
        'import pathlib,subprocess,sys,time; '
        'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]); '
        'pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)')
    with resources.dedicated_cgroup(root) as scope:
        with pytest.raises(subprocess.TimeoutExpired):
            resources.run_with_deadline(
                [sys.executable, '-c', script,
                 str(pid_file)], scope, 1)
    descendant = int(pid_file.read_text())
    for _ in range(50):
        try:
            if psutil.Process(descendant).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(.01)
    else:
        pytest.fail('The descendant survived the case deadline')


def test_parent_finalization_counts_exit_cpu_and_final_artifacts(
        tmp_path, monkeypatch):
    resources = load_resources()
    root = cgroup_fixture(resources, tmp_path, monkeypatch)
    output, scratch = tmp_path / 'output', tmp_path / 'scratch'
    output.mkdir()
    scratch.mkdir()
    record = dict(measured(), last_sample=dict(cpu_seconds=.1, monotonic=49.9))
    record['peaks']['cpu_cores'] = .5
    (output / 'result.json').write_bytes(b'x' * 8192)
    with resources.dedicated_cgroup(root) as scope:
        record['memory_accounting']['path'] = str(scope)
        (scope / 'memory.peak').write_text('400')
        (scope / 'cpu.stat').write_text('usage_usec 250000')
        monkeypatch.setattr(resources, 'time',
                            SimpleNamespace(monotonic=lambda: 50.))
        result = resources.finalize_resources(record, output, scratch, scope,
                                              3.5, completed=True)
        assert result['status'] == 'MEASURED'
        assert result['peaks']['ram_bytes'] == 400
        assert result['cpu_seconds'] == .25
        assert result['peaks']['cpu_cores'] == pytest.approx(1.5)
        assert result['peaks']['scratch_bytes'] >= 8192
        assert result['seconds'] == 3.5
        assert json.loads((output / 'resources.json').read_text()) == result


def test_timeout_retains_last_checkpoint_and_final_cgroup_counters(
        tmp_path, monkeypatch):
    resources = load_resources()
    root = cgroup_fixture(resources, tmp_path, monkeypatch)
    output, scratch = tmp_path / 'output', tmp_path / 'scratch'
    output.mkdir()
    scratch.mkdir()
    runtime = dict(gpu='observed GPU', gpu_count=1)
    monkeypatch.setattr(resources, '_gpu_memory_info', lambda: (80, 100))
    with resources.dedicated_cgroup(root) as scope:
        monkeypatch.setattr(resources, '_current_cgroup', lambda: scope)
        with resources.track_resources(output, scratch, runtime, scope):
            checkpoint = json.loads((output / 'resources.json').read_text())
        (output / 'resources.json').write_text(json.dumps(checkpoint))
        (scope / 'memory.peak').write_text('257')
        (scope / 'cpu.stat').write_text('usage_usec 500000')
        monkeypatch.setattr(resources, '_current_cgroup', lambda: root)
        result = resources.finalize_resources(None, output, scratch, scope, 30,
                                              completed=False,
                                              error='Case deadline exceeded')
        assert result['status'] == 'INCOMPLETE'
        assert result['peaks']['ram_bytes'] == 257
        assert result['cpu_seconds'] == .5
        assert result['peaks']['gpu_memory_bytes'] == 20
        assert any('deadline' in error for error in result['errors'])


def test_missing_worker_record_cannot_qualify_but_keeps_kernel_peaks(
        tmp_path, monkeypatch):
    resources = load_resources()
    root = cgroup_fixture(resources, tmp_path, monkeypatch)
    output, scratch = tmp_path / 'output', tmp_path / 'scratch'
    output.mkdir()
    scratch.mkdir()
    with resources.dedicated_cgroup(root) as scope:
        (scope / 'memory.peak').write_text('512')
        (scope / 'cpu.stat').write_text('usage_usec 45000')
        result = resources.finalize_resources(None, output, scratch, scope, 2,
                                              completed=False)
        assert result['status'] == 'INCOMPLETE'
        assert result['peaks']['ram_bytes'] == 512
        assert result['cpu_seconds'] == .045
        with pytest.raises(ValueError):
            resources.resource_requirements(result)
