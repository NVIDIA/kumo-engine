# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    cursor = SimpleNamespace(execute=lambda *args: calls.append(args),
                             fetchall=lambda: [])
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


@pytest.mark.parametrize('operation', ['prepare_payload', 'cache_lookup'])
def test_interrupted_upload_cleans_only_the_owned_submission_prefix(
        recovery, monkeypatch, tmp_path, operation):
    module, connection, calls = recovery
    path = tmp_path / 'report.json'
    report = json.loads(path.read_text())
    report.pop('retention')
    report.update(cleanup_owned=True, operation=operation)
    path.write_text(json.dumps(report))
    monkeypatch.setattr(module.retention, 'query_proof',
                        lambda *args: pytest.fail('No query was submitted'))
    assert module.recover(connection, tmp_path)['status'] == 'passed'
    assert calls[0] == ('cleanup',
                        dict(query_unknown=False, bundle_created=False,
                             bundle_pending=False))
    assert calls[1][0].startswith('LIST ')


@pytest.mark.parametrize('missing', [True, False])
def test_missing_submission_phase_cannot_authorize_cleanup(
        recovery, monkeypatch, tmp_path, missing):
    module, connection, calls = recovery
    path = tmp_path / 'report.json'
    report = json.loads(path.read_text())
    report.pop('retention')
    report.update(cleanup_owned=True)
    if not missing:
        report['operation'] = None
    path.write_text(json.dumps(report))
    monkeypatch.setattr(module.retention, 'query_proof', lambda *args:
                        pytest.fail('Missing phase cannot establish submission state'))
    assert module.recover(connection, tmp_path) == dict(
        status='preserved', reason='ownership_unconfirmed')
    assert calls == []


@pytest.mark.parametrize('remote', ['owned', 'different'])
@pytest.mark.parametrize('already_cleaned', [False, True])
def test_recovery_after_ledger_upload_before_report_update(
        recovery, monkeypatch, tmp_path, remote, already_cleaned):
    module, connection, calls = recovery
    path = tmp_path / 'report.json'
    report = json.loads(path.read_text())
    report.pop('retention')
    report.update(cleanup_owned=True, operation='cache_lookup')
    path.write_text(json.dumps(report))
    if already_cleaned:
        report['cleanup'] = dict(status='passed')
        path.write_text(json.dumps(report))
    present = [True]
    cursor = connection.cursor().__enter__()
    cursor.fetchall = lambda: [('ledger', )] if present[0] else []
    execute = cursor.execute

    def record(sql, *args):
        execute(sql, *args)
        if sql.startswith('REMOVE '):
            present[0] = False

    cursor.execute = record

    def download(cursor, source, destination):
        value = json.loads((tmp_path / 'retention-ledger.json').read_text())
        if remote == 'different':
            value['created_unix'] -= 1
        target = destination / 'retention-ledger.json'
        target.write_text(json.dumps(value))
        return target

    monkeypatch.setattr(module.launch, 'download_file', download)
    if remote == 'different':
        with pytest.raises(ValueError,
                           match='Remote unsubmitted ledger differs'):
            module.recover(connection, tmp_path)
        assert present[0]
    else:
        assert module.recover(connection, tmp_path)['status'] == 'passed'
        assert not present[0]
        assert module.recover(connection, tmp_path)['status'] == 'passed'
    assert sum(call[0].startswith('REMOVE ')
               for call in calls) == (remote == 'owned')


@pytest.mark.parametrize('operation,bundle', [('prepare_bundle', False),
                                              ('bundle_created', True)])
def test_recorded_ledger_before_execution_intent_recovers_owned_resources(
        recovery, monkeypatch, tmp_path, operation, bundle):
    module, connection, calls = recovery
    report_path = tmp_path / 'report.json'
    report = json.loads(report_path.read_text())
    report.update(cleanup_owned=True, operation=operation)
    report_path.write_text(json.dumps(report))
    cursor = connection.cursor().__enter__()
    cursor.fetchall = lambda: [('ledger', )]

    def download(cursor, location, destination):
        path = destination / 'retention-ledger.json'
        path.write_bytes((tmp_path / 'retention-ledger.json').read_bytes())
        return path

    monkeypatch.setattr(module.launch, 'download_file', download)
    monkeypatch.setattr(
        module.retention, 'query_proof',
        lambda *args: pytest.fail('Durable phase precedes execution intent'))
    inspected = []
    monkeypatch.setattr(
        module.retention, 'inspect',
        lambda *args: inspected.append(args) or {'owner_role': 'test-role'})
    assert module.recover(connection, tmp_path)['status'] == 'passed'
    cleanup = next(call for call in calls if call[0] == 'cleanup')
    assert cleanup[1] == dict(query_unknown=False, bundle_created=bundle,
                              bundle_pending=False)
    assert len(inspected) == (2 if bundle else 0)
    assert sum(call[0].startswith('REMOVE ') for call in calls) == 1


@pytest.mark.parametrize('operation', ['create_bundle', 'execute_notebook'])
def test_pending_intent_without_query_proof_preserves_resources(
        recovery, monkeypatch, tmp_path, operation):
    module, connection, calls = recovery
    path = tmp_path / 'report.json'
    report = json.loads(path.read_text())
    report.update(cleanup_owned=True, operation=operation)
    path.write_text(json.dumps(report))

    def missing(*args):
        raise ValueError('Missing or ambiguous notebook query history')

    monkeypatch.setattr(module.retention, 'query_proof', missing)
    with pytest.raises(ValueError, match='Missing or ambiguous'):
        module.recover(connection, tmp_path)
    assert calls == []


@pytest.mark.parametrize('changed', ['ledger', 'bundle'])
def test_known_unsubmitted_phase_preserves_changed_ownership(
        recovery, monkeypatch, tmp_path, changed):
    module, connection, calls = recovery
    path = tmp_path / 'report.json'
    report = json.loads(path.read_text())
    report.update(cleanup_owned=True, operation='bundle_created')
    path.write_text(json.dumps(report))
    cursor = connection.cursor().__enter__()
    cursor.fetchall = lambda: [('ledger', )]

    def download(cursor, location, destination):
        value = json.loads((tmp_path / 'retention-ledger.json').read_text())
        if changed == 'ledger':
            value['created_unix'] -= 1
        path = destination / 'retention-ledger.json'
        path.write_text(json.dumps(value))
        return path

    monkeypatch.setattr(module.launch, 'download_file', download)
    snapshots = iter([{'owner_role': 'owned'}, {'owner_role': 'changed'}])
    monkeypatch.setattr(module.retention, 'inspect',
                        lambda *args: next(snapshots))
    with pytest.raises(ValueError, match='differs|changed'):
        module.recover(connection, tmp_path)
    assert not any(call[0] == 'cleanup' or call[0].startswith('REMOVE ')
                   for call in calls)


@pytest.mark.parametrize('damaged', ['{"status":', '[]', '{"cleanup":[]}'])
def test_corrupt_journal_does_not_disable_sibling_recovery(
        monkeypatch, tmp_path, damaged):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    module = importlib.import_module('recover')
    for name in ('preflight', 'lifecycle'):
        (tmp_path / name).mkdir()
    (tmp_path / 'preflight/report.json').write_text(damaged)
    (tmp_path / 'lifecycle/report.json').write_text('{"status":"running"}')
    monkeypatch.setattr('sys.argv',
                        ['recover', '--output-dir',
                         str(tmp_path), '--execute'])
    monkeypatch.setattr('resources.require', lambda: None)
    monkeypatch.setattr(module.launch, 'connect', lambda key: nullcontext(None))
    recovered = []
    monkeypatch.setattr(
        module, 'recover', lambda connection, folder: recovered.append(
            folder.name) or dict(status='passed'))
    module.main()
    result = json.loads((tmp_path / 'recovery.json').read_text())
    assert recovered == ['lifecycle']
    assert result == dict(
        preflight=dict(status='preserved', reason='journal_unavailable'),
        lifecycle=dict(status='passed'))


@pytest.mark.parametrize('failure', [OSError, KeyboardInterrupt])
def test_failed_atomic_journal_replace_preserves_previous_json(
        monkeypatch, tmp_path, failure):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    module = importlib.import_module('launch')
    path = tmp_path / 'report.json'
    old = dict(cleanup_owned=True, operation='bundle_created')
    module.save(path, old)

    def interrupted(source, destination):
        assert json.loads(
            Path(source).read_text())['operation'] == 'execute_notebook'
        raise failure('interrupted before replacing journal')

    monkeypatch.setattr(module.os, 'replace', interrupted)
    with pytest.raises(failure):
        module.save(path, dict(old, operation='execute_notebook'))
    assert json.loads(path.read_text()) == old
    assert list(tmp_path.iterdir()) == [path]
