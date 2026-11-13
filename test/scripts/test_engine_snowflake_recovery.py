# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'packages/kumo-engine/test/snowflake_notebook'


@pytest.fixture
def recovery(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    module = importlib.import_module('recover')
    targets = module.retention.resources('a' * 12)
    plan = dict(
        attempt='a' * 12, bundle=targets['bundle'],
        stage_prefix=targets['submission'].rstrip('/'), mode='lifecycle',
        execute_sql='EXECUTE CODE BUNDLE ' + targets['bundle'] +
        " ENTRYPOINT = 'lifecycle.ipynb'")
    value = module.retention.ledger(plan)
    (tmp_path / 'report.json').write_text(
        json.dumps(dict(plan=plan, retention=dict(status='recorded'))))
    (tmp_path / 'retention-ledger.json').write_text(json.dumps(value))
    calls = []
    cursor = SimpleNamespace(execute=lambda *args: calls.append(args))
    connection = SimpleNamespace(cursor=lambda: nullcontext(cursor),
                                 get_query_status=lambda query: 'SUCCESS',
                                 is_still_running=lambda status: False)
    monkeypatch.setattr(module.retention, 'terminal_proof', lambda *args: 100)
    monkeypatch.setattr(module.retention, 'inspect',
                        lambda *args: {'owner': 'test-role'})
    monkeypatch.setattr(
        module.launch, 'cleanup_submission',
        lambda *args, **kwargs: calls.append(
            ('cleanup', kwargs)) or dict(status='passed'))
    return module, connection, calls


def test_recovery_cancels_only_the_proven_owned_query(recovery, monkeypatch,
                                                      tmp_path):
    module, connection, calls = recovery
    monkeypatch.setattr(
        module.retention, 'query_proof',
        lambda *args: dict(query_id='owned-query', execution_status='RUNNING'))
    assert module.recover(connection, tmp_path)['status'] == 'passed'
    assert calls[0] == ('SELECT SYSTEM$CANCEL_QUERY(%s)', ('owned-query', ))
    assert calls[-1][0] == 'cleanup' and calls[-1][1]['query_unknown'] is False


def test_recovery_preserves_unproven_query(recovery, monkeypatch, tmp_path):
    module, connection, calls = recovery

    def unknown(*args):
        raise ValueError('Missing or ambiguous query ownership')

    monkeypatch.setattr(module.retention, 'query_proof', unknown)
    with pytest.raises(ValueError, match='ownership'):
        module.recover(connection, tmp_path)
    assert calls == []


def test_recovery_preserves_query_that_does_not_stop(recovery, monkeypatch,
                                                     tmp_path):
    module, connection, calls = recovery
    monkeypatch.setattr(
        module.retention, 'query_proof',
        lambda *args: dict(query_id='owned-query', execution_status='RUNNING'))
    connection.is_still_running = lambda status: True
    times = iter([0, 31])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(times))
    assert module.recover(connection, tmp_path)['status'] == 'preserved'
    assert len(calls) == 1


def test_engine_workflow_always_attempts_bounded_recovery():
    workflow = yaml.safe_load(
        (ROOT /
         '.github/workflows/kumo-engine-snowflake-notebook.yml').read_text())
    step = next(step for step in workflow['jobs']['engine-lifecycle']['steps']
                if step.get('name') == 'Recover owned notebook submissions')
    assert step['if'] == 'always()' and step['timeout-minutes'] == 5
    assert 'recover.py' in step['run'] and '--execute' in step['run']


def test_completed_submissions_need_no_recovery_connection(
        monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    module = importlib.import_module('recover')
    folder = tmp_path / 'lifecycle'
    folder.mkdir()
    (folder / 'report.json').write_text(
        json.dumps(dict(cleanup=dict(status='passed'))))
    monkeypatch.setattr('sys.argv',
                        ['recover', '--output-dir',
                         str(tmp_path), '--execute'])
    monkeypatch.setattr(module.launch, 'connect',
                        lambda key: pytest.fail('No connection needed'))
    module.main()


def test_interrupted_upload_cleans_only_the_owned_submission_prefix(
        recovery, monkeypatch, tmp_path):
    module, connection, calls = recovery
    path = tmp_path / 'report.json'
    report = json.loads(path.read_text())
    report.pop('retention')
    report.update(cleanup_owned=True, operation='cache_lookup')
    path.write_text(json.dumps(report))
    monkeypatch.setattr(module.retention, 'query_proof',
                        lambda *args: pytest.fail('No query was submitted'))
    assert module.recover(connection, tmp_path)['status'] == 'passed'
    assert calls == [('cleanup',
                      dict(query_unknown=False, bundle_created=False,
                           bundle_pending=False))]
