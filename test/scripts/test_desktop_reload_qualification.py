# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/desktop_reload_qualification.py')
SPEC = importlib.util.spec_from_file_location('desktop_reload', SCRIPT)
RELOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELOAD)


def test_worker_environment(tmp_path, monkeypatch):
    for name in ('KUMO_CACHE', 'DISKGRAPH_BINARY'):
        monkeypatch.setenv(name, 'masked')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    env = RELOAD.worker_environment(tmp_path, 'cpu', {'DRQ_HIDE_FAISS': '1'})
    assert not any(value == 'masked' for value in env.values())
    for name in ('HOME', 'USERPROFILE'):
        assert env[name] == str(tmp_path / 'home')
    for name in ('TMPDIR', 'TEMP', 'TMP'):
        assert env[name] == str(tmp_path / 'tmp')
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    assert env['DRQ_HIDE_FAISS'] == '1'


def test_native_windows_label(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    assert RELOAD.label('cpu').startswith('windows-native-cpu-py')


def test_windows_timeout_terminates_tree(monkeypatch):
    process = Mock(pid=123)
    process.poll.return_value = None
    run = Mock()
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(subprocess, 'run', run)
    RELOAD.finish_process(process)
    run.assert_called_once_with(
        ['taskkill', '/PID', '123', '/T', '/F'],
        check=True, capture_output=True, timeout=30)
    process.wait.assert_called_once_with(timeout=30)


def test_windows_completed_process_is_reaped(monkeypatch):
    process = Mock()
    process.poll.return_value = 0
    run = Mock()
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(subprocess, 'run', run)
    RELOAD.finish_process(process)
    run.assert_not_called()
    process.wait.assert_called_once_with(timeout=30)


@pytest.mark.parametrize('exited', [True, False])
def test_windows_taskkill_exit_race(monkeypatch, exited):
    process = Mock(pid=123)
    process.poll.side_effect = [None, 0 if exited else None]
    run = Mock(side_effect=subprocess.CalledProcessError(128, 'taskkill'))
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(subprocess, 'run', run)
    if exited:
        RELOAD.finish_process(process)
        process.wait.assert_called_once_with(timeout=30)
    else:
        with pytest.raises(subprocess.CalledProcessError):
            RELOAD.finish_process(process)


@pytest.mark.parametrize('host,device,expected', [
    ('win32', 'cpu', 'auto'), ('linux', 'cpu', 'cpu'),
    ('win32', 'gpu', 'gpu'), ('darwin', 'mps', 'mps'),
])
def test_worker_accelerator(monkeypatch, host, device, expected):
    monkeypatch.setattr(sys, 'platform', host)
    assert RELOAD.worker_accelerator(device) == expected


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX process group check')
def test_spawn_isolates_then_removes_storage(tmp_path, monkeypatch):
    child = tmp_path / 'child.py'
    child.write_text('''import json, os, sys
from pathlib import Path
out, base = Path(sys.argv[6]), Path(sys.argv[7])
out.with_suffix('.json').write_text(json.dumps({
    'pid': os.getpid(), 'base': str(base),
    'masked': [key for key in ('KUMO_CACHE', 'DISKGRAPH_BINARY')
               if key in os.environ],
    'temp': os.environ['TEMP'], 'home': os.environ['USERPROFILE']}))
''')
    monkeypatch.setattr(RELOAD, '__file__', str(child))
    monkeypatch.setenv('KUMO_CACHE', 'masked')
    monkeypatch.setenv('DISKGRAPH_BINARY', 'masked')
    out = tmp_path / 'produce'
    assert RELOAD.spawn('produce', 'binary', 'cpu', tmp_path, out) == 0
    record = json.loads(out.with_suffix('.json').read_text())
    assert record['pid'] != os.getpid()
    assert record['masked'] == []
    assert record['temp'] == str(Path(record['base']) / 'tmp')
    assert record['home'] == str(Path(record['base']) / 'home')
    assert not Path(record['base']).exists()


@pytest.mark.parametrize('cleanup_error', [False, True])
def test_spawn_removes_storage_after_timeout(tmp_path, monkeypatch, cleanup_error):
    """A real worker times out; termination errors still reach storage cleanup."""
    if sys.platform == 'win32':
        pytest.skip('POSIX process-group timeout; Windows command tested above')
    child = tmp_path / 'child.py'
    child.write_text('import time; time.sleep(60)')
    monkeypatch.setattr(RELOAD, '__file__', str(child))
    launch = subprocess.Popen
    finish = RELOAD.finish_process
    processes = []
    bases = []

    def start(*args, **kwargs):
        process = launch(*args, **kwargs)
        processes.append(process)
        bases.append(Path(kwargs['cwd']).parent)
        wait = process.wait
        # Shorten the real first wait, retaining the cleanup's bounded reap.
        first_wait = True

        def bounded_wait(timeout):
            nonlocal first_wait
            limit = .05 if first_wait else timeout
            first_wait = False
            return wait(timeout=limit)

        process.wait = bounded_wait
        return process

    def terminate(process):
        finish(process)
        if cleanup_error:
            raise RuntimeError('termination failed')

    monkeypatch.setattr(subprocess, 'Popen', start)
    monkeypatch.setattr(RELOAD, 'finish_process', terminate)
    error = RuntimeError if cleanup_error else subprocess.TimeoutExpired
    with pytest.raises(error):
        RELOAD.spawn('produce', 'binary', 'cpu', tmp_path, tmp_path / 'produce')
    assert processes[0].poll() is not None
    assert not bases[0].exists()


def test_spawn_retains_live_worker_storage(tmp_path, monkeypatch):
    base = tmp_path / 'worker'
    base.mkdir()
    original = base / 'keep.txt'
    original.write_text('worker state')
    monkeypatch.setattr(RELOAD.tempfile, 'mkdtemp', lambda prefix: str(base))
    process = Mock()
    process.wait.side_effect = subprocess.TimeoutExpired('worker', 3600)
    process.poll.return_value = None
    monkeypatch.setattr(subprocess, 'Popen', Mock(return_value=process))
    error = subprocess.TimeoutExpired('taskkill', 30)
    monkeypatch.setattr(RELOAD, 'finish_process', Mock(side_effect=error))

    with pytest.raises(subprocess.TimeoutExpired) as failure:
        RELOAD.spawn('produce', 'binary', 'cpu', tmp_path, tmp_path / 'produce')

    assert failure.value is error
    assert original.read_text() == 'worker state'
    assert (base / 'cwd').is_dir()
