# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline identity/evidence guards; live notebook execution is qualified separately."""
import base64
import importlib
import json
from pathlib import Path
from urllib.parse import quote

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'packages/kumo-engine/test/databricks_notebook'


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    monkeypatch.syspath_prepend(str(DIRECTORY.parent))
    return importlib.import_module('databricks_notebook.lifecycle_notebook')


def passed_report():
    return dict(
        identity='kumo-databricks@nvidia.com', wheel_sha256='a' * 64,
        pip_check_returncode=0, phases={
            n:
            dict(
                status='passed', returncode=0,
                environment=dict(sha256='a' * 64), platform=dict(placement=dict(
                    context='databricks', user='kumo-databricks@nvidia.com')))
            for n in ('fit', 'score', 'evaluate', 'negative')
        }, required_gaps={'native_ui': dict(status='blocked')})


def test_engine_success_preserves_full_qualification_gap(runner):
    result = runner.engine_verdict(passed_report())
    assert result['engine_checks'] == 'passed'
    assert result['full_qualification'] == 'incomplete'
    assert result['remaining_requirements']['native_ui']['status'] == 'blocked'


@pytest.mark.parametrize('field', ['identity', 'wheel', 'context', 'pip'])
def test_passed_phase_requires_real_admission_evidence(runner, field):
    report = passed_report()
    phase = report['phases']['fit']
    if field == 'identity':
        phase['platform']['placement']['user'] = 'another-identity'
    elif field == 'context':
        phase['platform']['placement']['context'] = 'client'
    elif field == 'wheel':
        phase['environment']['sha256'] = 'b' * 64
    else:
        report['pip_check_returncode'] = 1
    with pytest.raises(ValueError):
        runner.engine_verdict(report)


def test_permission_block_does_not_qualify_all_engine_checks(runner):
    report = passed_report()
    report['phases']['negative'] = dict(status='blocked')
    result = runner.engine_verdict(report)
    assert result['positive_lifecycle'] == 'passed'
    assert result['engine_checks'] == 'incomplete'


@pytest.fixture
def evidence(runner, tmp_path):
    spec = dict(
        attempt='a' * 12, source_commit='b' * 40,
        candidate=dict(wheel_sha256='c' * 64), assets={}, test_sources={},
        output='/Volumes/kumo_test_catalogue/kumo_test/test_stage/test-only',
        notebook_path='/Shared/test-only')
    report = dict(status='blocked', attempt=spec['attempt'],
                  source_commit=spec['source_commit'],
                  wheel_sha256=spec['candidate']['wheel_sha256'], payload={},
                  test_sources={}, notebook_path=spec['notebook_path'],
                  identity='kumo-databricks@nvidia.com', execution_id='d' * 12)
    content = json.dumps(report).encode()
    receipt = dict(attempt=spec['attempt'], source_commit=spec['source_commit'],
                   output=spec['output'], execution_id=report['execution_id'],
                   files={'lifecycle-result.json': runner.digest(content)})
    submitted = runner.notebook(spec)
    commands = [
        dict(command=''.join(c['source']), state='finished',
             results=dict(data=[])) for c in submitted['cells']
    ]
    commands[-1]['results']['data'] = [
        dict(type='ansi',
             data='KUMO_DATABRICKS_LIFECYCLE_RECEIPT=' + json.dumps(receipt))
    ]
    return runner, spec, report, receipt, content, submitted, commands, tmp_path


def collect(evidence):
    runner, spec, report, receipt, content, submitted, commands, root = evidence
    encoded = base64.b64encode(
        quote(json.dumps(dict(commands=commands))).encode()).decode()
    export = dict(views=[
        dict(
            type='NOTEBOOK',
            content="<script>var __DATABRICKS_NOTEBOOK_MODEL = '" + encoded +
            "';</script>")
    ])

    class Client:
        def api(self, path, **kwargs):
            if '/runs/export?' in path:
                return export
            assert path == '/api/2.0/fs/files' + spec[
                'output'] + '/lifecycle-result.json'
            return content

    return runner.collect(
        Client(),
        dict(
            state=dict(result_state='SUCCESS'), tasks=[
                dict(task_key='graph_notebook', run_id=1,
                     state=dict(result_state='SUCCESS'))
            ]), spec, submitted, root)


def test_retained_lifecycle_is_bound_to_actual_export_and_hashes(evidence):
    assert collect(evidence)['execution_id'] == 'd' * 12


def test_lifecycle_download_cannot_substitute_another_report(evidence):
    mutated = list(evidence)
    mutated[4] = b'changed bytes'
    with pytest.raises(ValueError, match='checksum mismatch'):
        collect(mutated)


def test_unrelated_cell_source_is_rejected(evidence):
    evidence[6][-1]['command'] = 'print("unrelated")'
    with pytest.raises(ValueError, match='submitted execution'):
        collect(evidence)


def test_nonzero_worker_exit_cannot_be_qualified_from_a_passed_receipt(runner):
    report = passed_report()
    report['phases']['negative']['returncode'] = 1
    with pytest.raises(ValueError, match='worker exit code'):
        runner.engine_verdict(report)


def test_dirty_lifecycle_source_is_rejected_before_auth_or_cloud_calls(
        runner, tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(
        runner.subprocess, 'check_output', lambda command, **kwargs: 'a' * 40
        if 'rev-parse' in command else b'different committed source')
    monkeypatch.setattr(
        runner, 'Client',
        lambda config: pytest.fail('No cloud client should be constructed'))
    with pytest.raises(ValueError,
                       match='Commit the exact lifecycle test inputs'):
        runner.execute(SimpleNamespace(cluster_id='0101-010101-test'), tmp_path)


def test_notebook_keeps_committed_kernel_after_working_source_changes(
        runner, monkeypatch):
    def no_read(path, *args, **kwargs):
        raise AssertionError(
            'Frozen notebook construction must not reread working source')

    monkeypatch.setattr(Path, 'read_text', no_read)
    value = runner.notebook(dict(attempt='a' * 12),
                            kernel_source='COMMITTED_KERNEL_SENTINEL = True')
    code = ''.join(value['cells'][-1]['source'])
    assert 'COMMITTED_KERNEL_SENTINEL = True' in code
