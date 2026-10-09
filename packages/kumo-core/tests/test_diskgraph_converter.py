# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from types import SimpleNamespace

import pytest
from kumo_core.diskgraph import converter


def test_converter_subprocess_environment_does_not_mutate_parent(
        monkeypatch, tmp_path):
    captured = {}

    def run(command, **kwargs):
        captured['command'] = command
        captured['environment'] = kwargs['env']
        result_index = command.index('--result') + 1
        with open(command[result_index], 'w') as result_file:
            json.dump({'tables': []}, result_file)
        return SimpleNamespace(returncode=0, stderr='')

    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'parent-key')
    monkeypatch.setenv('AWS_SESSION_TOKEN', 'parent-token')
    monkeypatch.setattr(converter.subprocess, 'run', run)

    result = converter.convert_diskgraph_config(
        {'name': 'graph'},
        str(tmp_path),
        {'events': 'v1'},
        subprocess_environment={
            'AWS_ACCESS_KEY_ID': 'child-key',
            'AWS_SESSION_TOKEN': None,
        },
    )

    assert result == {'tables': []}
    assert captured['environment']['AWS_ACCESS_KEY_ID'] == 'child-key'
    assert 'AWS_SESSION_TOKEN' not in captured['environment']
    assert os.environ['AWS_ACCESS_KEY_ID'] == 'parent-key'
    assert os.environ['AWS_SESSION_TOKEN'] == 'parent-token'


def test_converter_without_environment_runs_in_process(monkeypatch):
    monkeypatch.setattr(converter, '_convert', lambda request: request)

    result = converter.convert_diskgraph_config(
        {'name': 'graph'},
        '/tmp/persist',
        {'events': 'v1'},
    )

    assert result['graph_config'] == {'name': 'graph'}
    assert result['table_data_versions'] == {'events': 'v1'}


def test_converter_subprocess_failure_separates_safe_and_restricted_detail(
        monkeypatch, tmp_path):
    canary = 'warehouse-row-value-must-require-debug'
    monkeypatch.setattr(
        converter.subprocess,
        'run',
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=17,
            stderr=canary,
        ),
    )

    with pytest.raises(converter.DiskgraphConfigConversionError) as exc_info:
        converter.convert_diskgraph_config(
            {'name': 'graph'},
            str(tmp_path),
            {'events': 'v1'},
            subprocess_environment={'AWS_ACCESS_KEY_ID': 'child-key'},
        )

    error = exc_info.value
    assert error._kumo_restricted_log_message is True
    assert error._kumo_safe_log_message == (
        'DiskGraph config conversion subprocess failed (exit code 17)')
    assert canary in str(error)
