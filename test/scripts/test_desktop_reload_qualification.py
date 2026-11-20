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

from io_utils import fsutil

SCRIPT = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/desktop_reload_qualification.py')
SPEC = importlib.util.spec_from_file_location('desktop_reload', SCRIPT)
RELOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELOAD)


def test_worker_environment(tmp_path, monkeypatch):
    for name in ('CI', 'GITHUB_ACTIONS', 'PYTEST_CURRENT_TEST',
                 'SYNC_DATA_LOADER', 'KUMO_CACHE', 'DISKGRAPH_BINARY'):
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


@pytest.mark.skipif(sys.platform == 'win32', reason='non-Windows refusal check')
def test_windows_cleanup_check_refuses_other_platforms():
    checker = SCRIPT.with_name('windows_cleanup_check.py')
    result = subprocess.run([sys.executable, '-I', str(checker)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'requires native Windows' in result.stderr
    evidence = json.loads(result.stdout.splitlines()[0])
    assert evidence['sys_platform'] == sys.platform
    assert evidence['os_name'] == os.name


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX process group check')
def test_spawn_isolates_then_removes_storage(tmp_path, monkeypatch):
    child = tmp_path / 'child.py'
    fsutil.write_text(str(child), '''import json, os, sys
from pathlib import Path
out, base = Path(sys.argv[6]), Path(sys.argv[7])
out.with_suffix('.json').write_text(json.dumps({
    'pid': os.getpid(), 'base': str(base),
    'masked': [key for key in ('GITHUB_ACTIONS', 'SYNC_DATA_LOADER',
                              'PYTEST_CURRENT_TEST') if key in os.environ],
    'temp': os.environ['TEMP'], 'home': os.environ['USERPROFILE']}))
''')
    monkeypatch.setattr(RELOAD, '__file__', str(child))
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('SYNC_DATA_LOADER', '1')
    out = tmp_path / 'produce'
    assert RELOAD.spawn('produce', 'binary', 'cpu', tmp_path, out) == 0
    record = json.loads(fsutil.read_text(str(out.with_suffix('.json'))))
    assert record['pid'] != os.getpid()
    assert record['masked'] == []
    assert record['temp'] == str(Path(record['base']) / 'tmp')
    assert record['home'] == str(Path(record['base']) / 'home')
    assert not fsutil.exists(record['base'])
