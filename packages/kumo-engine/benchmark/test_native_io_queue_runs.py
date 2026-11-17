# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
from native_io_queue_runs import (
    observer_command,
    observer_order,
    pin_source,
    ram_log,
    run_service,
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
    spec = {'timeout': 1, 'result': '/result', 'observer_paths': {'ready': '/ready'},
            'iops_paths': {'gate': '/gate', 'release': '/release', 'stop': '/stop',
                           'service_log': str(tmp_path / 'service.log'),
                           'trace_log': str(tmp_path / 'observer.log')}}
    args = argparse.Namespace(service_user='user', startup_timeout=1, finalize_timeout=300)
    with pytest.raises(RuntimeError, match='service stop failed'):
        run_service(spec, args, 'queue', {})
    observer.wait.assert_called_once_with(timeout=300)
    candidate.kill_collector.assert_not_called()
    candidate.publish_json.assert_any_call('/stop', {'exit_code': 1, 'result_path': '/result'})
    candidate.validate_result.assert_not_called()
