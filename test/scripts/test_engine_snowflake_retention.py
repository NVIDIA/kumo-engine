# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retention admission uses provider proof, not age or names alone."""
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[2] / 'packages/kumo-engine/test'
ATTEMPT = 'a' * 12
NOW = 1000000


@pytest.fixture
def retention(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY / 'snowflake_notebook'))
    return importlib.import_module('retention')


def date(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc)


def state(retention):
    targets = retention.resources(ATTEMPT)
    value = retention.ledger(
        dict(
            attempt=ATTEMPT, bundle=targets['bundle'],
            stage_prefix=targets['submission'].rstrip('/'), mode='lifecycle',
            execute_sql='EXECUTE CODE BUNDLE ' + targets['bundle'] +
            " ENTRYPOINT = 'lifecycle.ipynb'"), created=NOW - 3 * retention.TTL)
    query = dict(query_id='query-id', query_tag=value['query_tag'],
                 user_name=retention.launch.USER,
                 role_name=retention.launch.ROLE,
                 query_text=value['execute_sql'], execution_status='SUCCESS',
                 end_time=date(NOW - 2 * retention.TTL))
    obj = dict(name=targets['model'].rsplit('.',
                                            1)[1], owner=retention.launch.ROLE,
               database_name=retention.launch.DATABASE,
               schema_name=retention.launch.SCHEMA,
               created_on=date(value['created_unix'] + 1))
    return value, query, obj


class Cursor:
    def __init__(self, history, obj, bundle=None):
        self.bundle = bundle
        self.history = history
        self.obj = obj
        self.commands = []
        self.returned = []

    def execute(self, sql):
        self.commands.append(sql)
        if 'QUERY_HISTORY_BY_USER' in sql:
            returned = self.history
        elif sql.startswith('SHOW CODE BUNDLES'):
            returned = [self.bundle] if self.bundle else []
        elif sql.startswith('SHOW '):
            returned = [self.obj] if self.obj else []
        else:
            returned = []
        self.description = [(name, )
                            for name in returned[0]] if returned else []
        self.returned = [tuple(row.values()) for row in returned]
        return self

    def fetchall(self):
        return self.returned

    @property
    def mutations(self):
        return [
            sql for sql in self.commands
            if sql.startswith(('DROP ', 'REMOVE ', 'PUT '))
        ]


def test_plan_never_mutates_and_expire_removes_only_exact_resources(retention):
    value, query, obj = state(retention)
    cursor = Cursor([query], obj)
    assert retention.expire(cursor, value, now=NOW)['status'] == 'planned'
    assert not cursor.mutations
    result = retention.expire(cursor, value, execute=True, now=NOW)
    assert result['status'] == 'passed'
    target = retention.resources(ATTEMPT)
    assert cursor.mutations == [
        'DROP MODEL ' + target['model'],
        'DROP EXPERIMENT ' + target['experiment'],
        'REMOVE ' + target['submission'], 'REMOVE ' + target['artifacts'],
        'REMOVE ' + retention.CONTROL + ATTEMPT + '/'
    ]
    assert not any('payload-cache' in sql for sql in cursor.mutations)


@pytest.mark.parametrize('change', [
    'active', 'foreign_user', 'foreign_role', 'foreign_tag', 'different_query',
    'missing', 'ambiguous', 'recent_end', 'old_window', 'recent_attempt',
    'truncated'
])
def test_uncertain_query_always_preserves_all_resources(retention, change):
    value, query, obj = state(retention)
    history = [query]
    if change == 'active':
        query['execution_status'] = 'RUNNING'
    elif change == 'foreign_user':
        query['user_name'] = 'OTHER'
    elif change == 'foreign_role':
        query['role_name'] = 'OTHER'
    elif change == 'foreign_tag':
        query['query_tag'] = 'OTHER'
    elif change == 'different_query':
        query['query_text'] += ' changed'
    elif change == 'missing':
        history = []
    elif change == 'ambiguous':
        history *= 2
    elif change == 'recent_end':
        query['end_time'] = date(NOW - 10)
    elif change == 'old_window':
        value['created_unix'] = NOW - 7 * retention.TTL
    elif change == 'recent_attempt':
        value['created_unix'] = NOW - 10
    elif change == 'truncated':
        history *= 10000
    cursor = Cursor(history, obj)
    with pytest.raises(ValueError):
        retention.expire(cursor, value, execute=True, now=NOW)
    assert not cursor.mutations


@pytest.mark.parametrize(
    'change', ['owner', 'database_name', 'schema_name', 'before', 'after'])
def test_replaced_or_foreign_objects_preserve_entire_attempt(retention, change):
    value, query, obj = state(retention)
    if change in ('owner', 'database_name', 'schema_name'):
        obj[change] = 'OTHER'
    else:
        obj['created_on'] = date(value['created_unix'] -
                                 1 if change == 'before' else NOW)
    cursor = Cursor([query], obj)
    with pytest.raises(ValueError):
        retention.expire(cursor, value, execute=True, now=NOW)
    assert not cursor.mutations


@pytest.mark.parametrize('attempt', ['../shared', 'A' * 12, 'a' * 11, None])
def test_invalid_attempt_cannot_select_resource(retention, attempt):
    with pytest.raises(ValueError):
        retention.resources(attempt)


def test_shortened_retention_is_rejected(retention):
    value, query, obj = state(retention)
    cursor = Cursor([query], obj)
    with pytest.raises(ValueError, match='24 hours'):
        retention.expire(cursor, value, execute=True, now=NOW, ttl=1)
    assert not cursor.commands


def test_changed_object_between_plan_and_delete_is_preserved(
        retention, monkeypatch):
    value, query, obj = state(retention)
    cursor = Cursor([query], obj)
    original = retention.inspect
    counts = {}

    def inspect(*args):
        kind = args[1]
        counts[kind] = counts.get(kind, 0) + 1
        result = original(*args)
        if kind == 'MODEL' and counts[kind] == 2:
            return dict(result, created_on=date(NOW))
        return result

    monkeypatch.setattr(retention, 'inspect', inspect)
    with pytest.raises(ValueError, match='changed during cleanup'):
        retention.expire(cursor, value, execute=True, now=NOW)
    assert not cursor.mutations


def test_notebook_attempt_binding_does_not_change_other_platforms(
        monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    platform = importlib.import_module('wheel_mlops_platform')
    config = json.loads(
        (DIRECTORY / 'mlops_platforms/snowflake.json').read_text())
    config.update(
        schema='KUMO_TEST',
        artifact_root='@KUMO_TEST_DB.KUMO_TEST.TEST_STAGE/mlops-validation')
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(config))
    monkeypatch.setenv(platform.NOTEBOOK_ATTEMPT, ATTEMPT)
    platform.prepare(tmp_path, path)
    assert json.loads(
        (tmp_path / 'platform.json').read_text())['attempt'] == ATTEMPT
    monkeypatch.setenv(platform.NOTEBOOK_ATTEMPT, '../foreign')
    with pytest.raises(ValueError, match='attempt'):
        platform.prepare(tmp_path, path)


def test_old_unprovable_ledgers_cannot_starve_eligible_attempts(
        retention, monkeypatch):
    prefix = 'test_stage/mlops-validation/retention/'
    old = [
        dict(name=prefix + f'{i:012x}/retention-ledger.json',
             last_modified=date(NOW - 7 * retention.TTL)) for i in range(120)
    ]
    eligible = [
        dict(name=prefix + f'{i:012x}/retention-ledger.json',
             last_modified=date(NOW - 2 * retention.TTL))
        for i in range(200, 320)
    ]
    cursor = Cursor([], None)
    original_execute = cursor.execute

    def execute(sql):
        if sql.startswith('LIST '):
            cursor.description = [('name', ), ('last_modified', )]
            cursor.returned = [tuple(row.values()) for row in old + eligible]
        else:
            original_execute(sql)

    cursor.execute = execute
    cursor.__class__.__enter__ = lambda self: self
    cursor.__class__.__exit__ = lambda *args: None
    connection = type('Connection', (), {'cursor': lambda self: cursor})()
    seen = []

    def download(cursor, location, folder):
        attempt = location.split('/')[-2]
        file = folder / 'retention-ledger.json'
        file.write_text(json.dumps(dict(attempt=attempt)))
        return file

    monkeypatch.setattr(retention.time, 'time', lambda: NOW)
    monkeypatch.setattr(retention.launch, 'download_file', download)
    monkeypatch.setattr(
        retention, 'expire',
        lambda cursor, value, **kw: seen.append(value['attempt']))
    result = retention.sweep(connection, execute=True)
    assert result['expired'] == len(seen) == 100
    assert result['outside_history'] == 120
    assert all(int(attempt, 16) >= 200 for attempt in seen)


def test_durable_ledger_failure_blocks_notebook_submission(
        retention, tmp_path, monkeypatch):
    target = retention.resources(ATTEMPT)
    manifest = dict(
        attempt=ATTEMPT, bundle=target['bundle'],
        stage_prefix=target['submission'].rstrip('/'),
        execute_sql='EXECUTE CODE BUNDLE ' + target['bundle'] +
        " ENTRYPOINT = 'probe.ipynb'")
    cursor = Cursor([], None)
    monkeypatch.setattr(retention.launch, 'download_file',
                        lambda *a: pytest.fail('GET after failed PUT'))
    with pytest.raises(RuntimeError, match='upload failed'):
        retention.write(cursor, tmp_path, manifest)
    assert len(cursor.mutations) == 1
    assert cursor.mutations[0].startswith('PUT ')
    assert 'OVERWRITE=FALSE' in cursor.mutations[0]


def test_ledger_admission_rejects_foreign_bundle(retention):
    target = retention.resources(ATTEMPT)
    with pytest.raises(ValueError, match='non-attempt'):
        retention.ledger(
            dict(attempt=ATTEMPT, bundle='PROD.PUBLIC.MODEL',
                 stage_prefix=target['submission'].rstrip('/')))


@pytest.mark.parametrize('prefix', ['submission', 'artifacts'])
def test_retention_never_removes_a_pinned_shared_archive(
        retention, monkeypatch, prefix):
    value, query, obj = state(retention)
    cursor = Cursor([query], obj)
    location = retention.resources(
        value['attempt'])[prefix] + 'payload/wheelhouse.tar'
    monkeypatch.setattr(retention, 'protected_locations', lambda: (location, ))
    with pytest.raises(ValueError, match='protected shared cache'):
        retention.expire(cursor, value, execute=True, now=NOW)
    assert not cursor.commands


@pytest.mark.parametrize('status', [
    'SUCCESS', 'FAILED_WITH_ERROR', 'FAILED_WITH_INCIDENT', 'success',
    'failed_with_error', 'failed_with_incident'
])
def test_terminal_status_uses_provider_case_insensitively(retention, status):
    value, query, obj = state(retention)
    query['execution_status'] = status
    assert retention.expire(Cursor([query], obj), value,
                            now=NOW)['status'] == 'planned'


@pytest.mark.parametrize('status', ['RUNNING', 'QUEUED', 'BLOCKED', None, 1])
def test_nonterminal_or_malformed_provider_status_is_preserved(
        retention, status):
    value, query, obj = state(retention)
    query['execution_status'] = status
    cursor = Cursor([query], obj)
    with pytest.raises(ValueError, match='terminal query identity'):
        retention.expire(cursor, value, execute=True, now=NOW)
    assert not cursor.mutations


def bundle_metadata(retention, value):
    return dict(
        name=retention.resources(value['attempt'])['bundle'].rsplit('.', 1)[-1],
        database_name=retention.launch.DATABASE,
        schema_name=retention.launch.SCHEMA,
        created_on=date(value['created_unix'] + 1),
        updated_on=date(value['created_unix'] + 2),
        owner_role=retention.launch.ROLE, owner_role_type='ROLE', comment=None)


def test_code_bundle_uses_live_owner_role_metadata_shape(retention):
    value, query, obj = state(retention)
    bundle = bundle_metadata(retention, value)
    cursor = Cursor([query], obj, bundle=bundle)
    retention.expire(cursor, value, execute=True, now=NOW)
    assert cursor.mutations[0] == 'DROP CODE BUNDLE ' + retention.resources(
        ATTEMPT)['bundle']


@pytest.mark.parametrize('change', [
    'different_owner', 'missing_owner', 'legacy_owner', 'database_role',
    'missing_role_type', 'unknown_role_type'
])
def test_code_bundle_ownership_cannot_be_guessed(retention, change):
    value, query, obj = state(retention)
    bundle = bundle_metadata(retention, value)
    if change == 'different_owner':
        bundle['owner_role'] = 'OTHER_ROLE'
    elif change == 'missing_owner':
        bundle.pop('owner_role')
    elif change == 'legacy_owner':
        bundle['owner'] = bundle.pop('owner_role')
    elif change == 'database_role':
        bundle['owner_role_type'] = 'DATABASE_ROLE'
    elif change == 'missing_role_type':
        bundle.pop('owner_role_type')
    else:
        bundle['owner_role_type'] = 'UNKNOWN'
    cursor = Cursor([query], obj, bundle=bundle)
    with pytest.raises(ValueError, match='ownership'):
        retention.expire(cursor, value, execute=True, now=NOW)
    assert not cursor.mutations


def test_local_plan_accepts_cached_key_file(retention, monkeypatch, tmp_path):
    from contextlib import nullcontext

    import resources

    key = tmp_path / 'cached-key.p8'
    observed = []
    connection = object()
    monkeypatch.setattr(resources, 'require', lambda: {})
    monkeypatch.setattr(
        retention.launch, 'connect',
        lambda value: observed.append(value) or nullcontext(connection))

    def plan(selected, **options):
        assert selected is connection
        assert options['execute'] is False
        return {'status': 'planned'}

    monkeypatch.setattr(retention, 'sweep', plan)
    monkeypatch.setattr('sys.argv', ['retention.py', '--key-file', str(key)])
    retention.main()
    assert observed == [key]


def test_sweep_continues_after_provider_cleanup_error(retention, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    prefix = 'test_stage/mlops-validation/retention/'
    attempts = ['a' * 12, 'b' * 12]
    cursor = SimpleNamespace(
        description=[('name', ), ('last_modified', )], execute=lambda sql: None,
        fetchall=lambda: [(prefix + attempt + '/retention-ledger.json',
                           date(NOW - 2 * retention.TTL))
                          for attempt in attempts])
    connection = SimpleNamespace(cursor=lambda: nullcontext(cursor))

    def download(cursor, location, folder):
        file = folder / 'retention-ledger.json'
        file.write_text(json.dumps(dict(attempt=location.split('/')[-2])))
        return file

    visited = []

    def expire(cursor, value, **kwargs):
        visited.append(value['attempt'])
        if len(visited) == 1:
            raise OSError('Provider deletion unavailable')

    monkeypatch.setattr(retention.time, 'time', lambda: NOW)
    monkeypatch.setattr(retention.launch, 'download_file', download)
    monkeypatch.setattr(retention, 'expire', expire)
    report = retention.sweep(connection, execute=True)
    assert set(visited) == set(attempts)
    assert report['expired'] == report['preserved'] == report[
        'cleanup_errors'] == 1
