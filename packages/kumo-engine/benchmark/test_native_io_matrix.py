# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Counter and matched-work checks that prevent misleading benchmark results."""

from collections import Counter

import native_io_resources
import pytest
from native_io_matrix import (
    BACKENDS,
    confidence_interval,
    reuse_trial,
    runner_command,
    summarize,
    training_evidence,
    trial_order,
    validate_native,
    validate_work,
)
from native_io_resources import (
    evict_files,
    owned_files,
    parse_counters,
    parse_io,
    residency,
    warm_files,
)

from io_utils import fsutil


def result(backend: str, repeat: int, seconds: float) -> dict:
    return dict(task='binary', regime='resident', backend=backend,
                repeat=repeat, status='PASS', config_sha256='fixed',
                predictions=dict(values=[[.1, .9]], decisions=[True]),
                provenance=dict(native_sha256='a' * 64),
                config=dict(loader_mode='standalone'),
                work=dict(optimizer_steps=128, train_batches=128,
                          predict_rows=1000, predict_batches=4),
                phase_seconds=dict(fit=seconds, predict=seconds / 10,
                                   e2e=seconds * 1.1))


def test_balanced_order():
    ordered = trial_order(5, 12)
    assert ordered == trial_order(5, 12)
    assert set(ordered) == {(repeat, backend) for repeat in range(5)
                            for backend in BACKENDS}
    for position in range(3):
        counts = Counter(backend for _, backend in ordered[position::3])
        assert max(counts.values()) - min(counts.values()) <= 1


def test_work_mismatch_rejected():
    rows = [result('mmap', 0, 20), result('direct', 0, 10)]
    rows[1]['work']['optimizer_steps'] = 127
    with pytest.raises(ValueError, match='different work'):
        validate_work(rows)


def test_config_mismatch_rejected():
    rows = [result('mmap', 0, 20), result('direct', 0, 10)]
    rows[1]['config_sha256'] = 'different'
    with pytest.raises(ValueError, match='different model'):
        validate_work(rows)


def test_incomplete_matrix_rejected():
    with pytest.raises(ValueError, match='Incomplete matrix'):
        summarize([result('mmap', 0, 20)], 5, 12)


def test_paired_speedups():
    rows = [result(backend, repeat, (repeat + 10) /
                   (2 if backend == 'direct' else 1))
            for repeat in range(5) for backend in BACKENDS]
    summary = summarize(rows, 5, 12)
    direct = summary['groups']['binary/resident']['direct']['e2e']
    assert direct['paired_speedups'] == [2.] * 5
    assert direct['median_speedup'] == 2.
    assert direct['speedup_bootstrap_95'] == [2., 2.]
    assert len(summary['trials']) == 15


def test_wrong_predictions_rejected():
    rows = [result(backend, 0, 10) for backend in BACKENDS]
    rows[1]['predictions']['values'][0][0] = .2
    with pytest.raises(ValueError, match='tolerance'):
        summarize(rows, 1, 12)


def test_bootstrap_reproducible():
    values = [.8, .9, 1, 1.1, 1.2]
    assert confidence_interval(values, 12) == confidence_interval(values, 12)
    with pytest.raises(ValueError, match='finite and positive'):
        confidence_interval([float('nan')], 12)


def test_cgroup_counters():
    assert parse_counters('anon 100\nfile 500\n') == dict(anon=100, file=500)
    assert parse_io('259:0 rbytes=2048 wbytes=1024 rios=2\n'
                    '259:1 rbytes=4096 wbytes=0 rios=1\n') == dict(
                        rbytes=6144, wbytes=1024, rios=3)


def test_owned_files_boundary(tmp_path):
    with pytest.raises(ValueError, match='outside prepared'):
        owned_files(dict(owned_files=['/etc/passwd']), str(tmp_path))
    with pytest.raises(ValueError, match='no owned files'):
        owned_files(dict(owned_files=[]), str(tmp_path))


def test_cache_control(tmp_path):
    path = str(tmp_path / 'owned-index.bin')
    fsutil.write_binary(path, b'x' * 65536)
    assert warm_files([path])['resident_fraction'] == 1.
    assert residency(path) == (16, 16)
    assert evict_files([path])['resident_pages'] == 0
    assert residency(path) == (0, 16)


def test_missing_io_is_unknown(monkeypatch):
    values = {
        'memory.current': '100', 'memory.peak': '200', 'memory.max': '1024',
        'memory.swap.max': '0', 'memory.stat': 'anon 50\nfile 50',
        'memory.events': 'oom 0\noom_kill 0', 'cpu.stat': 'usage_usec 5',
    }
    monkeypatch.setattr(fsutil, 'exists', lambda path: False)
    monkeypatch.setattr(fsutil, 'read_text',
                        lambda path: values[path.rsplit('/', 1)[1]])
    monkeypatch.setattr(native_io_resources, 'process_snapshot',
                        lambda: dict(pid=1, scope='process_only'))
    snapshot = native_io_resources.group_snapshot('/scope')
    assert snapshot['io_available'] is False
    assert snapshot['io'] is None
    assert snapshot['memory_max'] == '1024'


def test_cache_off_rejected():
    row = result('direct', 0, 10)
    row['provenance']['environment'] = {'DG_DIRECT_CACHE_GB': '1'}
    row['native_io'] = {'fit': {'after': dict(
        configured_backend='direct', runtime_initialized=True, io_ops=1,
        runtime_rings=2, cache_initialized=False, cache_capacity_pages=0)}}
    spec = dict(backend='direct', direct_cache_gib=1, io_rings=2,
                loader_mode='standalone')
    with pytest.raises(ValueError, match='cache capacity'):
        validate_native(row, spec)
    row['native_io']['fit']['after'].update(
        cache_initialized=True, cache_capacity_pages=262144)
    validate_native(row, spec)


@pytest.mark.parametrize('changed', ('steps', 'loader_mode'))
def test_resume_requires_matching_work(changed):
    fields = ('task', 'backend', 'regime', 'repeat', 'prepared_dir', 'memory_bytes',
              'steps', 'batch_size', 'workers', 'threads', 'torch_threads',
              'model_seed', 'python', 'runner', 'direct_cache_gib', 'io_rings',
              'io_qd', 'cpus', 'loader_mode', 'prepared_manifest_sha256')
    spec = {field: 1 for field in fields}
    row = result('direct', 0, 10)
    row['matrix_spec'] = {**spec, changed: 2}
    with pytest.raises(ValueError, match='parameters differ'):
        reuse_trial(row, spec, 'a' * 64)


def test_index_size_alone_is_insufficient():
    def snapshot(read_bytes, scan):
        return dict(observer_process=dict(pid=1, io=dict(read_bytes=read_bytes)),
                    memory_stat=dict(pgscan=scan))

    row = result('direct', 0, 10)
    row['provenance']['index_bytes'] = 100
    row['work']['training_interval'] = dict(
        seconds=60, resources_before=snapshot(0, 0),
        resources_after=snapshot(4096, 0),
        native_before=dict(evictions=0), native_after=dict(evictions=0))
    spec = dict(workers=0, memory_bytes=40)
    assert not training_evidence(row, spec)['has_pressure_evidence']
    row['work']['training_interval']['resources_after'] = snapshot(4096, 1)
    assert training_evidence(row, spec)['has_pressure_evidence']


def test_loader_mode_forwarded():
    fields = ('python', 'runner', 'task', 'backend', 'prepared_dir', 'result',
              'steps', 'batch_size', 'workers', 'torch_threads', 'threads',
              'model_seed', 'repeat', 'regime')
    spec = {field: '1' for field in fields}
    spec['loader_mode'] = 'standalone'
    command = runner_command(spec)
    assert command[command.index('--loader-mode') + 1] == 'standalone'
    spec['loader_mode'] = 'auto'
    command = runner_command(spec)
    assert command[command.index('--loader-mode') + 1] == 'auto'
