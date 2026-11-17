# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
from native_io_queue_runs import (
    durable_artifacts,
    observer_command,
    observer_order,
    pin_source,
    quiet_device,
    ram_log,
    run_service,
    validate_controller,
    validate_identity,
)

from io_utils import fsutil


def test_balanced_pair_identity():
    rows = observer_order(5, ['mmap', 'direct'], ['none', 'queue'])
    assert len(rows) == len(set(rows)) == 20
    assert rows == observer_order(5, ['mmap', 'direct'], ['none', 'queue'])
    for repeat in range(5):
        for backend in ('mmap', 'direct'):
            assert {mode for rep, kind, mode in rows if (rep, kind) == (repeat, backend)} == {
                'none', 'queue'}


@pytest.mark.parametrize('repeats,backends,modes', [
    (0, ['mmap'], ['queue']), (1, [], ['queue']), (1, ['mmap'], []),
    (1, ['mmap', 'mmap'], ['queue']), (1, ['mmap'], ['none', 'none']),
])
def test_reject_duplicate_or_empty_cells(repeats, backends, modes):
    with pytest.raises(ValueError):
        observer_order(repeats, backends, modes)


def test_ready_cgroup_and_controller_are_required():
    gate = {'pid': 101, 'cgroup_id': 202, 'cgroup_path': '/scope'}
    ready = {'status': 'READY', 'gate_pid': 101, 'cgroup_id': 202, 'cgroup_path': '/scope'}
    validate_identity(ready, gate)
    for key, value in [('gate_pid', 102), ('cgroup_id', 203), ('cgroup_path', '/other')]:
        with pytest.raises(ValueError):
            validate_identity({**ready, key: value}, gate)


def test_observer_commands_preserve_venv_and_full_payload():
    spec = {'python': '/isolated/venv/bin/python', 'timeout': 3600,
            'iops_paths': {'gate': '/gate', 'stop': '/stop'},
            'observer_paths': {'ready': '/ready', 'record': '/record', 'events': '/events'}}
    args = argparse.Namespace(collector_cpus='0-4,10-14', candidate_root='/candidate',
                              device='/dev/nvme0n1', drain_seconds=10,
                              bcc_python_path='/bcc', request_capacity=65536,
                              event_capacity_bytes=536870912, quiet_seconds=1, frequency=99)
    queue = observer_command(spec, args, 'queue')
    assert queue[:6] == ['taskset', '-c', '0-4,10-14', 'sudo', '-n', spec['python']]
    assert queue[queue.index('--candidate-root') + 1] == '/candidate'
    assert queue[queue.index('--request-capacity') + 1] == '65536'
    assert '--steps' not in queue
    for mode in ('cpu', 'offcpu'):
        command = observer_command(spec, args, mode)
        assert command[command.index('--mode') + 1] == mode
        assert '--request-capacity' not in command
    with pytest.raises(ValueError):
        observer_command(spec, args, 'none')


def test_source_pin_rejects_changes(monkeypatch):
    responses = iter(['correct', ' M tracked.py'])
    monkeypatch.setattr('native_io_queue_runs.git', lambda *args: next(responses))
    with pytest.raises(ValueError, match='clean'):
        pin_source('/repo', 'correct')


def test_observer_log_is_not_published_until_stop(tmp_path):
    destination = tmp_path / 'observer.log'
    with ram_log(str(destination)) as stream:
        # A child writes the inherited fd, as sudo/collectors do in production.
        subprocess.run(['/usr/bin/printf', 'observer output\n'], stdout=stream, check=True)
        assert not fsutil.exists(str(destination))
    assert fsutil.read_binary(str(destination)) == b'observer output\n'


def test_failed_observer_always_publishes_stop_and_reaps(tmp_path, monkeypatch):
    candidate = ModuleType('native_io_iops')
    candidate.phase_rates = Mock()
    candidate.publish_json = Mock()
    candidate.service_command = lambda spec, unit, user: ['payload', unit, user]
    candidate.collector_group_alive = Mock(return_value=False)
    candidate.kill_collector = Mock()
    candidate.stop_service = Mock(side_effect=RuntimeError('service stop failed'))
    candidate.validate_result = Mock()
    gate = {'pid': 10, 'cgroup_id': 20, 'cgroup_path': '/scope'}
    ready = {'status': 'READY', 'gate_pid': 10, 'cgroup_id': 20, 'cgroup_path': '/scope'}
    candidate.wait_json = lambda path, process, timeout: gate if path == '/gate' else ready
    monkeypatch.setitem(sys.modules, 'native_io_iops', candidate)
    service, observer = Mock(), Mock()
    service.poll.return_value = None
    observer.poll.side_effect = [None, 7]
    monkeypatch.setattr('native_io_queue_runs.subprocess.Popen', Mock(side_effect=[service, observer]))
    monkeypatch.setattr('native_io_queue_runs.observer_command', lambda *args: ['observer'])
    monkeypatch.setattr('native_io_queue_runs.durable_artifacts', Mock())
    monkeypatch.setattr('native_io_queue_runs.quiet_device', Mock())
    spec = {'timeout': 1, 'result': '/result', 'observer_paths': {'ready': '/ready'},
            'iops_paths': {'gate': '/gate', 'release': '/release', 'stop': '/stop',
                           'service_log': str(tmp_path / 'service.log'),
                           'trace_log': str(tmp_path / 'observer.log')}}
    args = argparse.Namespace(service_user='user', startup_timeout=1, finalize_timeout=300,
                              output_dir=str(tmp_path), controller_log=None,
                              device='/device', quiet_seconds=1, quiet_timeout=30)
    with pytest.raises(RuntimeError, match='service stop failed'):
        run_service(spec, args, 'queue', {})
    observer.wait.assert_called_once_with(timeout=300)
    candidate.kill_collector.assert_not_called()
    candidate.publish_json.assert_any_call('/stop', {'exit_code': 1, 'result_path': '/result'})
    candidate.validate_result.assert_not_called()


def test_durability_covers_owned_files_and_cleanup_directories(tmp_path, monkeypatch):
    root = str(tmp_path / 'experiment')
    mode = root + '/none'
    prepared = mode + '/trial-prepared'
    fsutil.makedirs(prepared)
    result = mode + '/trial.json'
    for path in (result, mode + '/trial.log', prepared + '/manifest.json', root + '/progress.json'):
        fsutil.write_text(path, 'owned')
    outside = str(tmp_path / 'baseline')
    fsutil.write_text(outside, 'unchanged')
    spec = {'result': result, 'iops_paths': {}, 'observer_paths': {},
            'runtime_prepared_dir': prepared}
    descriptors = []
    monkeypatch.setattr('native_io_queue_runs.os.fsync',
                        lambda fd: descriptors.append(os.readlink(f'/proc/self/fd/{fd}')))
    synced = durable_artifacts(spec, root)
    assert {result, mode + '/trial.log', prepared + '/manifest.json', root + '/progress.json'} <= set(synced)
    assert outside not in descriptors
    assert descriptors[-3:] == [prepared, mode, root]
    fsutil.rm(prepared)
    descriptors.clear()
    durable_artifacts(spec, root)
    assert descriptors[-2:] == [mode, root]


def test_durability_fails_on_external_path_and_sync_error(tmp_path, monkeypatch):
    root = str(tmp_path / 'owned')
    fsutil.makedirs(root)
    spec = {'result': str(tmp_path / 'outside.json'), 'iops_paths': {},
            'observer_paths': {}, 'runtime_prepared_dir': root + '/prepared'}
    with pytest.raises(ValueError, match='restricted'):
        durable_artifacts(spec, root)
    spec['result'] = root + '/result.json'
    fsutil.write_text(spec['result'], 'owned')
    monkeypatch.setattr('native_io_queue_runs.os.fsync', Mock(side_effect=OSError('sync failed')))
    with pytest.raises(OSError, match='sync failed'):
        durable_artifacts(spec, root)


def fake_device(monkeypatch, values):
    trace = ModuleType('native_io_trace')
    trace.device_identity = lambda path: {'path': path, 'sysfs': '/physical'}
    monkeypatch.setitem(sys.modules, 'native_io_trace', trace)
    clock = [0.]
    monkeypatch.setattr('native_io_queue_runs.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('native_io_queue_runs.time.monotonic_ns', lambda: int(clock[0] * 1e9))
    monkeypatch.setattr('native_io_queue_runs.time.sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    pending = list(values)

    def read(path):
        assert path == '/physical/stat'
        if len(pending) > 1:
            return pending.pop(0)
        return pending[0]

    monkeypatch.setattr('native_io_queue_runs.fsutil.read_text', read)


def test_quiet_requires_stable_device_counters_and_zero_inflight(monkeypatch):
    fake_device(monkeypatch, ['0 0 0 0 0 0 0 0 1 0 0',
                              '1 0 8 0 0 0 0 0 0 0 0', '2 0 16 0 0 0 0 0 0 0 0'])
    proof = quiet_device('/dev/physical', .1, 1)
    assert proof['device']['path'] == '/dev/physical'
    assert proof['ended_ns'] - proof['quiet_started_ns'] >= 100_000_000
    assert proof['stable_counters'][0] == 2


def test_quiet_timeout_and_bad_statistics_reject(monkeypatch):
    fake_device(monkeypatch, ['0 0 0 0 0 0 0 0 1 0 0'])
    with pytest.raises(TimeoutError, match='quiet'):
        quiet_device('/dev/physical', .1, .2)
    fake_device(monkeypatch, ['0 0 0'])
    with pytest.raises(ValueError, match='statistics'):
        quiet_device('/dev/physical', .1, 1)


def test_controller_log_requires_exact_owned_stdout(tmp_path, monkeypatch):
    root = str(tmp_path / 'experiment')
    path = root + '-controller.log'
    fsutil.write_text(path, '')
    monkeypatch.setattr('native_io_queue_runs.os.fstat', lambda fd: os.stat(path))
    validate_controller(root, path)
    with pytest.raises(ValueError, match='requires'):
        validate_controller(root, None)
    with pytest.raises(ValueError, match='exact'):
        validate_controller(root, str(tmp_path / 'unrelated.log'))


@pytest.mark.parametrize('mode', ['none', 'queue'])
def test_quiet_failure_never_releases_either_payload(tmp_path, monkeypatch, mode):
    candidate = ModuleType('native_io_iops')
    candidate.phase_rates = Mock()
    candidate.publish_json = Mock()
    candidate.service_command = lambda spec, unit, user: ['payload']
    candidate.stop_service = Mock()
    candidate.validate_result = Mock()
    gate = {'pid': 10, 'cgroup_id': 20, 'cgroup_path': '/scope'}
    ready = {'status': 'READY', 'gate_pid': 10, 'cgroup_id': 20, 'cgroup_path': '/scope'}
    candidate.wait_json = lambda path, process, timeout: gate if path == '/gate' else ready
    monkeypatch.setitem(sys.modules, 'native_io_iops', candidate)
    service, observer = Mock(), Mock()
    service.poll.return_value = 0
    observer.poll.return_value = None
    monkeypatch.setattr('native_io_queue_runs.subprocess.Popen', Mock(side_effect=[service, observer]))
    monkeypatch.setattr('native_io_queue_runs.observer_command', lambda *args: ['observer'])
    durability = Mock()
    monkeypatch.setattr('native_io_queue_runs.durable_artifacts', durability)

    def failed_quiet(*args):
        durability.assert_called_once()
        raise TimeoutError('device busy')

    monkeypatch.setattr('native_io_queue_runs.quiet_device', failed_quiet)
    monkeypatch.setattr('native_io_queue_runs.finish_observer', Mock())
    spec = {'timeout': 1, 'result': '/result', 'observer_paths': {'ready': '/ready'},
            'iops_paths': {'gate': '/gate', 'release': '/release', 'stop': '/stop',
                           'service_log': str(tmp_path / 'service.log'),
                           'trace_log': str(tmp_path / 'observer.log')}}
    args = argparse.Namespace(service_user='user', startup_timeout=1, finalize_timeout=300,
                              output_dir=str(tmp_path), controller_log=None,
                              device='/device', quiet_seconds=1, quiet_timeout=30)
    with pytest.raises(TimeoutError, match='device busy'):
        run_service(spec, args, mode, {})
    assert [call.args[0] for call in candidate.publish_json.call_args_list] == ['/stop']
    candidate.stop_service.assert_called_once()
