# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared notebook sessions at the provider's read and write boundaries."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kumo_core.sources import principals
from kumo_snowflake import session as sessions
from kumo_snowflake.backend import SnowflakeSourceBackend
from kumo_snowflake.sink import SnowflakeSinkBackend
from kumoapi.data_source import (
    CreateIdentityArgs,
    SnowflakeAuthenticationKind,
    SnowflakeConnectorResourceConfig,
    SnowflakeIdentityConfig,
)
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session


@pytest.fixture
def notebook(monkeypatch):
    pytest.importorskip('snowflake.snowpark.mock')
    monkeypatch.setattr(principals, 'store', principals.PrincipalStore())
    monkeypatch.setattr(sessions, '_sessions', sessions._SessionCache())
    session = Session.builder.config('local_testing', True).create()
    # Local testing has no Session.sql implementation; context resolution must
    # use tracked getters without issuing SQL or closing the caller's session.
    with monkeypatch.context() as checks:
        checks.setattr(session, 'sql', Mock(side_effect=AssertionError('SQL')))
        checks.setattr(session, 'close',
                       Mock(side_effect=AssertionError('close')))
        yield session
    session.close()


def connector(notebook, **overrides):
    target = dict(warehouse=notebook.get_current_warehouse(),
                  role=notebook.get_current_role(),
                  database=notebook.get_current_database(),
                  schema_name=notebook.get_current_schema())
    target.update(overrides)
    return SimpleNamespace(id='notebook', identity=None, identity_id=None,
                           _snowpark_session_hint='Select a session explicitly.',
                           config=SnowflakeConnectorResourceConfig(**target))


def test_ambient_session_is_shared_without_context_changes(notebook):
    target = connector(notebook, warehouse='mock_warehouse', database='OTHER',
                       schema_name='OTHER')
    for backend in (SnowflakeSourceBackend(), SnowflakeSinkBackend()):
        with backend._session(target) as found:
            assert found is notebook
    sessions.close()
    assert get_active_session() is notebook
    notebook.sql.assert_not_called()
    notebook.close.assert_not_called()


@pytest.mark.parametrize('field', ['warehouse', 'role'])
def test_shared_target_is_checked_on_every_resolution(notebook, field):
    target = connector(notebook)
    backend = SnowflakeSourceBackend()
    with backend._session(target) as found:
        assert found is notebook
    target.config = replace(target.config, **{field: '"OTHER"'})
    with pytest.raises(ValueError) as error:
        with backend._session(target):
            pass
    assert str(error.value) == (
        f"Connector {field} 'OTHER' differs from the shared Snowpark session "
        f"{field} 'MOCK_{field.upper()}'. Switch the session {field}, or "
        'supply an identity for a separate session.')


@pytest.mark.parametrize('field', ['database', 'schema'])
@pytest.mark.parametrize('missing', [False, True])
def test_sql_source_refuses_a_different_shared_context(
        notebook, monkeypatch, field, missing):
    config_field = 'schema_name' if field == 'schema' else field
    target = connector(notebook, **{config_field: '"OTHER"'})
    if missing:
        monkeypatch.setattr(notebook, f'get_current_{field}', lambda: None)
    with pytest.raises(ValueError) as error:
        SnowflakeSourceBackend().get_columns(target, 'query',
                                             sql='SELECT 1 AS value')
    if missing:
        reason = (f'The shared Snowpark session has no current {field}; '
                  f"SQL sources require the connector {field} 'OTHER'.")
    else:
        reason = (f'SQL sources require the shared session {field} '
                  f"'MOCK_{field.upper()}' to match the connector "
                  f"{field} 'OTHER'.")
    assert str(error.value) == (
        f'{reason} Call session.use_database(...) and session.use_schema(...).')
    notebook.sql.assert_not_called()


@pytest.mark.parametrize('explicit', [True, False])
def test_identity_precedes_the_shared_session(notebook, monkeypatch, explicit):
    registration = principals.store.create(CreateIdentityArgs(
        display_name='session-test', config=SnowflakeIdentityConfig(
            account='account',
            authentication=SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE)))
    target = connector(notebook)
    target._snowpark_session = notebook
    if explicit:
        target.identity_id = registration.slug
    backend = SnowflakeSourceBackend(
        identity=None if explicit else registration.slug)
    own = object()
    get = Mock(return_value=own)
    monkeypatch.setattr(sessions._sessions, 'get', get)
    with backend._session(target) as found:
        assert found is own
    assert get.call_args.args == (registration.principal, target)


def test_an_unresolved_identity_does_not_fall_back_to_shared_session(notebook):
    target = connector(notebook)
    target.identity_id = 'missing-notebook-identity'
    target._snowpark_session = notebook
    with pytest.raises(
            ValueError, match="names 'missing-notebook-identity'.*none exists"):
        with SnowflakeSourceBackend()._session(target):
            pass


def test_ambient_resolution_cannot_choose_between_sessions(notebook):
    other = Session.builder.config('local_testing', True).create()
    try:
        target = connector(notebook)
        with pytest.raises(ValueError, match='More than one.*Select a session explicitly'):
            with SnowflakeSourceBackend()._session(target):
                pass
        target._snowpark_session = notebook
        with SnowflakeSourceBackend()._session(target) as found:
            assert found is notebook
    finally:
        other.close()


def test_kumo_owned_sessions_are_not_ambient(notebook, monkeypatch):
    monkeypatch.setattr(sessions, '_authentication', lambda _: {})
    monkeypatch.setattr(sessions.Session.SessionBuilder, 'create',
                        lambda _: notebook)
    with sessions._owned_session(SimpleNamespace(name='own'), None):
        with pytest.raises(ValueError,
                           match='No Snowflake identity or shared'):
            with SnowflakeSourceBackend()._session(connector(notebook)):
                pass
    # _owned_session owns this test session; its close must be attempted.
    assert notebook.close.call_count == 1


def test_shared_write_propagates_token_expiry_without_reconnect(
        notebook, monkeypatch, tmp_path):
    import pandas as pd

    class ExpiredTokenError(Exception):
        errno = 390114
        sqlstate = '08001'

    path = tmp_path / 'part.parquet'
    pd.DataFrame({'id': [1]}).to_parquet(path)
    error = ExpiredTokenError()
    export = Mock(side_effect=error)
    connect = Mock(side_effect=AssertionError('reconnect'))
    monkeypatch.setattr('kumo_snowflake.write.export_table_files', export)
    monkeypatch.setattr(sessions, '_connect', connect)
    with pytest.raises(ExpiredTokenError) as caught:
        SnowflakeSinkBackend().write_files(connector(notebook), 'OUT',
                                           [str(path)])
    assert caught.value is error
    assert export.call_count == 1
    connect.assert_not_called()
    notebook.close.assert_not_called()


def test_stage_preflight_consumes_the_named_show_result(notebook, monkeypatch):
    from kumo_snowflake.artifacts import StageArtifactLocation

    show = Mock(return_value=SimpleNamespace(collect=lambda: [{
        'created_on': 'unused', 'name': 'STAGE', 'database_name': 'DB',
        'schema_name': 'SCHEMA', 'type': 'EXTERNAL', 'extra_column': 'unused',
    }]))
    monkeypatch.setattr(notebook, 'sql', show)
    registration = principals.store.create(CreateIdentityArgs(
        display_name='stage-test', config=SnowflakeIdentityConfig(
            account='account',
            authentication=SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE)))
    monkeypatch.setattr(sessions._sessions, 'get', lambda *_: notebook)
    # artifacts imports the principal store directly.
    monkeypatch.setattr('kumo_snowflake.artifacts.identity_store',
                        principals.store)
    monkeypatch.setattr('kumo_snowflake.artifacts._sessions', sessions._sessions)
    with pytest.raises(ValueError, match='requires a named internal stage'):
        location = StageArtifactLocation('@DB.SCHEMA.STAGE', registration.slug)
        location.preflight()
    show.assert_called_once_with('SHOW STAGES IN SCHEMA DB.SCHEMA')


@pytest.mark.parametrize('code', ['1403', '1409'])
def test_missing_hosted_identity_keeps_the_hosted_remedy(
        notebook, monkeypatch, code):
    from snowflake.snowpark.exceptions import SnowparkSessionException

    target = connector(notebook)
    del target._snowpark_session_hint
    error = SnowparkSessionException('No unique session', error_code=code)
    monkeypatch.setattr(sessions, 'get_active_session', Mock(side_effect=error))
    with pytest.raises(ValueError, match='kumoai_studio.create_identity') as error:
        with SnowflakeSourceBackend()._session(target):
            pass
    assert 'names no identity' in str(error.value)
    assert 'from_session' not in str(error.value)


@pytest.mark.parametrize('operation', ['columns', 'sample', 'ingest'])
def test_sql_context_is_checked_on_the_session_acquired_for_the_read(
        notebook, monkeypatch, tmp_path, operation):
    target = connector(notebook)
    backend = SnowflakeSourceBackend()
    embeddable = backend._embeddable

    def parse_then_switch(connector, sql):
        query = embeddable(connector, sql)
        monkeypatch.setattr(notebook, 'get_current_database',
                            lambda: '"OTHER"')
        return query

    monkeypatch.setattr(backend, '_embeddable', parse_then_switch)
    query = 'SELECT 1 AS value'
    with pytest.raises(ValueError, match='SQL sources require.*OTHER'):
        if operation == 'columns':
            backend.get_columns(target, 'query', sql=query)
        elif operation == 'sample':
            backend.get_sample(target, 'query', 1, sql=query)
        else:
            backend.prepare_ingest(target, 'query', str(tmp_path), sql=query)
    notebook.sql.assert_not_called()
