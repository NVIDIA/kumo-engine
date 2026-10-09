# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""New local tests and fixtures must work without editing the public runner."""

import sys
from unittest.mock import patch

import pytest

from scripts import test_public_connectors


@pytest.mark.parametrize('keyword, expected',
                         [(None, 1), ('keep', 0), ('fail', 1), ('missing', 5)])
def test_focused_file_preserves_fixtures_and_exit_codes(
        tmp_path, monkeypatch, keyword, expected):
    (tmp_path / 'conftest.py').write_text(
        'import pytest\n@pytest.fixture\ndef value(): return 42\n')
    selected = tmp_path / 'test_selected.py'
    selected.write_text('def test_keep(value): assert value == 42\n'
                        'def test_fail(): assert False\n')
    (tmp_path / 'test_other.py').write_text(
        'raise RuntimeError("Unselected file was imported")\n')
    monkeypatch.setitem(test_public_connectors.SUITES, 'csv', (str(tmp_path), ))
    arguments = ['test_public_connectors.py', '--test-file', str(selected)]
    if keyword is not None:
        arguments += ['-k', keyword]
    monkeypatch.setattr(sys, 'argv', arguments)
    try:
        result = test_public_connectors.main()
    except SystemExit as error:
        result = error.code
    assert result == expected


def test_keyword_filter_keeps_cloud_modules_unloaded(tmp_path, monkeypatch):
    (tmp_path / 'test_local.py').write_text(
        'def test_keep(): assert True\ndef test_drop(): assert False\n')
    for tier in ('live', 'native'):
        (tmp_path / tier).mkdir()
        (tmp_path / tier / 'test_cloud.py').write_text(
            'raise RuntimeError("Excluded module was imported")\n')
    monkeypatch.setitem(test_public_connectors.SUITES, 'csv', (str(tmp_path), ))
    monkeypatch.setattr(sys, 'argv',
                        ['test_public_connectors.py', '-k', 'keep and not drop'])
    try:
        result = test_public_connectors.main()
    except SystemExit as error:
        result = error.code
    assert result == 0


def test_focused_reader_run_requires_one_suite(monkeypatch):
    monkeypatch.setattr(sys, 'argv',
                        ['test_public_connectors.py', '--suite', 'all', '-k', 'keep'])
    with patch.object(test_public_connectors.subprocess, 'run') as run:
        with pytest.raises(SystemExit) as error:
            test_public_connectors.main()
        assert error.value.code == 2
        run.assert_not_called()


@pytest.mark.parametrize('case',
                         ['live', 'native', 'symlink', 'outside', 'directory', 'missing'])
def test_focused_file_cannot_select_excluded_paths(tmp_path, monkeypatch, case):
    suite = tmp_path / 'suite'
    suite.mkdir()
    cloud = suite / 'live' / 'test_cloud.py'
    cloud.parent.mkdir()
    cloud.write_text('raise RuntimeError("Live code must not be imported")\n')
    native = suite / 'native' / 'test_native.py'
    native.parent.mkdir()
    native.write_text('raise RuntimeError("Native code must not be imported")\n')
    alias = suite / 'test_alias.py'
    alias.symlink_to(cloud)
    outside = tmp_path / 'test_outside.py'
    outside.write_text('raise RuntimeError("Outside code must not be imported")\n')
    target = {'live': cloud, 'native': native, 'symlink': alias,
              'outside': outside, 'directory': suite,
              'missing': suite / 'test_missing.py'}[case]
    monkeypatch.setitem(test_public_connectors.SUITES, 'csv', (str(suite), ))
    monkeypatch.setattr(sys, 'argv',
                        ['test_public_connectors.py', '--test-file', str(target)])
    with patch.object(test_public_connectors.subprocess, 'run') as run:
        with pytest.raises(SystemExit) as error:
            test_public_connectors.main()
        assert error.value.code == 2
        run.assert_not_called()


def test_new_test_runs_with_local_setup_and_cloud_files_stay_unloaded(
        tmp_path, monkeypatch):
    suite = tmp_path / 'tests'
    suite.mkdir()
    (suite / 'conftest.py').write_text(
        'import pytest\n@pytest.fixture\ndef local_value(): return 42\n')
    (suite / 'test_existing.py').write_text(
        'def test_existing(local_value): assert local_value == 42\n')
    live = suite / 'live'
    live.mkdir()
    (live / 'test_cloud.py').write_text(
        'raise RuntimeError("Live code must not be imported")\n')
    monkeypatch.setitem(test_public_connectors.SUITES, 'csv', (str(suite),))
    monkeypatch.setattr(sys, 'argv', ['test_public_connectors.py'])
    assert test_public_connectors.main() == 0

    # No runner or image rule changes when a developer adds another test.
    added = suite / 'test_added.py'
    added.write_text('def test_added(local_value): assert local_value == 0\n')
    assert test_public_connectors.main() == 1
    added.write_text('def test_added(local_value): assert local_value == 42\n')
    assert test_public_connectors.main() == 0


def test_loopback_can_run_first_without_enabling_external_connections(
        tmp_path, monkeypatch):
    probe = tmp_path / 'test_socket_policy.py'
    probe.write_text('''
import socket
import pytest
import pytest_socket

# Collection must stay offline even before the first test starts.
with pytest.raises(pytest_socket.SocketBlockedError):
    socket.socket()

@pytest.mark.allow_hosts(["127.0.0.1"])
def test_first_loopback():
    with socket.socket() as server:
        server.settimeout(2)
        server.bind(("127.0.0.1", 0))
        server.listen()
        with socket.create_connection(server.getsockname(), timeout=2):
            connection, _ = server.accept()
            connection.close()
    with socket.socket() as client:
        with pytest.raises(pytest_socket.SocketConnectBlockedError):
            client.connect(("192.0.2.1", 80))

def test_unmarked_still_blocks_sockets():
    with pytest.raises(pytest_socket.SocketBlockedError):
        socket.socket()
''')
    monkeypatch.setitem(test_public_connectors.SUITES, 'csv', (str(tmp_path), ))
    monkeypatch.setattr(sys, 'argv', ['test_public_connectors.py'])
    assert test_public_connectors.main() == 0
