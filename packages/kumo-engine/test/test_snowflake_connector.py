# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from kumo_snowflake import spcs
from kumo_snowflake.spcs import ACCOUNT_ENV, HOST_ENV
from kumoapi.data_source import SnowflakeAuthenticationKind

from kumo.engine.connector.snowflake import SnowflakeConnector


def test_connection_toml_is_loaded_by_the_provider_package(monkeypatch):
    requested = []

    def connection_toml_entry(connection_name):
        requested.append(connection_name)
        return {
            'account': 'xy12345',
            'user': 'alice',
            'password': 'secret',
            'warehouse': 'WH',
            'database': 'DB',
            'schema': 'SCHEMA',
            'role': 'ANALYST',
        }

    monkeypatch.setattr(
        'kumo_snowflake.config.connection_toml_entry',
        connection_toml_entry,
    )

    connector = SnowflakeConnector.from_connection_toml(
        connection_name='development')

    assert requested == ['development']
    assert connector.warehouse == 'WH'
    assert connector.database == 'DB'
    assert connector.schema_name == 'SCHEMA'
    assert connector.role == 'ANALYST'
    assert (connector.identity._config.authentication is
            SnowflakeAuthenticationKind.PASSWORD)


@pytest.fixture
def container(monkeypatch, tmp_path):
    """A mounted token and the account/host SPCS exports beside it, plus the
    warehouse/database/schema it exports for the service's own target.
    """
    token = tmp_path / "token"
    token.write_text("ey.mounted.token")
    monkeypatch.setattr(spcs, "SPCS_TOKEN_PATH", str(token))
    monkeypatch.setenv(ACCOUNT_ENV, "xy12345")
    monkeypatch.setenv(HOST_ENV, "snowflake-host.internal")
    monkeypatch.setenv("SNOWFLAKE_WAREHOUSE", "WH")
    monkeypatch.setenv("SNOWFLAKE_DATABASE", "DB")
    monkeypatch.setenv("SNOWFLAKE_SCHEMA", "SCHEMA")
    return token


def test_the_container_environment_supplies_every_field(container):
    connector = SnowflakeConnector.from_spcs_env()

    assert connector.warehouse == "WH"
    assert connector.database == "DB"
    assert connector.schema_name == "SCHEMA"
    identity_config = connector.identity._config
    assert identity_config.account == "xy12345"
    assert identity_config.host == "snowflake-host.internal"
    assert identity_config.token_path == str(container)
    assert (identity_config.authentication is
            SnowflakeAuthenticationKind.OAUTH_TOKEN_FILE)


def test_warehouse_database_schema_are_overridable(container):
    connector = SnowflakeConnector.from_spcs_env(
        warehouse="OTHER_WH", database="OTHER_DB", schema_name="OTHER_SCHEMA")

    assert connector.warehouse == "OTHER_WH"
    assert connector.database == "OTHER_DB"
    assert connector.schema_name == "OTHER_SCHEMA"


def test_role_is_passed_through(container):
    connector = SnowflakeConnector.from_spcs_env(role="ANALYST")

    assert connector.role == "ANALYST"


def test_no_mounted_token_means_this_is_not_a_container(monkeypatch, tmp_path):
    monkeypatch.setattr(spcs, "SPCS_TOKEN_PATH", str(tmp_path / "absent"))
    monkeypatch.setenv(ACCOUNT_ENV, "xy12345")

    with pytest.raises(ValueError, match="not a snowpark container services"):
        SnowflakeConnector.from_spcs_env()


def test_a_token_without_an_account_is_reported_separately(
        container, monkeypatch):
    monkeypatch.delenv(ACCOUNT_ENV, raising=False)

    with pytest.raises(ValueError, match="is mounted but"):
        SnowflakeConnector.from_spcs_env()


def test_a_missing_target_field_with_no_override_is_reported(
        container, monkeypatch):
    monkeypatch.delenv("SNOWFLAKE_WAREHOUSE", raising=False)

    with pytest.raises(ValueError, match="names no warehouse"):
        SnowflakeConnector.from_spcs_env()


def test_session_is_the_readers_session(monkeypatch):
    from kumo_snowflake import session as sessions

    connector = SnowflakeConnector(warehouse='WH', database='DB',
                                   schema_name='S')
    calls = []
    monkeypatch.setattr(
        sessions._sessions, 'get',
        lambda principal, target, fresh=False: calls.append(target) or 'live')
    monkeypatch.setattr(type(connector.source_backend), '_principal',
                        lambda self, target, **kwargs: 'principal')
    assert connector.session() == 'live'
    assert calls == [connector]


@pytest.fixture
def notebook_session():
    pytest.importorskip('snowflake.snowpark.mock')
    from snowflake.snowpark import Session

    session = Session.builder.config('local_testing', True).create()
    try:
        yield session
    finally:
        session.close()


def test_from_session_reads_context_and_keeps_one_active_session(
        notebook_session):
    from snowflake.snowpark.context import get_active_session

    connector = SnowflakeConnector.from_session()
    assert connector.warehouse == 'MOCK_WAREHOUSE'
    assert connector.role == 'MOCK_ROLE'
    assert connector.database == 'MOCK_DATABASE'
    assert connector.schema_name == 'MOCK_SCHEMA'
    assert connector.identity is None
    assert connector.session() is notebook_session
    assert get_active_session() is notebook_session
    same_target = SnowflakeConnector(warehouse='MOCK_WAREHOUSE',
                                     database='MOCK_DATABASE',
                                     schema_name='MOCK_SCHEMA',
                                     role='MOCK_ROLE')
    assert same_target.id == connector.id


def test_from_session_selects_a_session_when_two_are_active(notebook_session):
    from snowflake.snowpark import Session

    other = Session.builder.config('local_testing', True).create()
    try:
        with pytest.raises(ValueError, match='More than one.*from_session'):
            SnowflakeConnector.from_session()
        connector = SnowflakeConnector.from_session(notebook_session)
        assert connector.session() is notebook_session
    finally:
        other.close()


def test_pickling_drops_the_session_in_a_fresh_process(
        notebook_session, tmp_path):
    import os
    import pickle
    import subprocess
    import sys

    connector = SnowflakeConnector.from_session(notebook_session)
    assert connector.session() is notebook_session
    payload = tmp_path / 'connector.pickle'
    payload.write_bytes(
        pickle.dumps((connector, connector.config, connector.id)))
    script = '''
import pickle
import sys
from pathlib import Path

connector, config, identifier = pickle.loads(Path(sys.argv[1]).read_bytes())
assert connector.config == config
assert connector.id == identifier
try:
    connector.session()
except ValueError as error:
    assert 'No Snowflake identity or shared Snowpark session' in str(error)
    assert 'from_session' in str(error) and 'identity=' in str(error), error
else:
    raise AssertionError('a fresh process has no shared session')
'''
    result = subprocess.run(
        [sys.executable, '-c', script, str(payload)], cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('name,expected', [
    ('"UPPER_CASE$1"', 'UPPER_CASE$1'),
    ('"Mixed"', '"Mixed"'),
    ('"123"', '"123"'),
    ('"WITH SPACE"', '"WITH SPACE"'),
    ('"A""B"', '"A""B"'),
])
def test_from_session_preserves_identifier_meaning(notebook_session, monkeypatch,
                                                   name, expected):
    for field in ('warehouse', 'role', 'database', 'schema'):
        monkeypatch.setattr(notebook_session, f'get_current_{field}',
                            lambda: name)
    connector = SnowflakeConnector.from_session(notebook_session)
    target = SnowflakeConnector(warehouse=expected, role=expected,
                                database=expected, schema_name=expected)
    assert connector.config == target.config
    assert connector.id == target.id
    assert connector.session() is notebook_session


def test_warehouse_mismatch_has_a_specific_remedy(notebook_session, monkeypatch):
    monkeypatch.setattr(notebook_session, 'get_current_warehouse',
                        lambda: '"KUMO_PYTORCH_USABILITY_WH"')
    connector = SnowflakeConnector(warehouse='KUMO_SHEET_CHECK_NO_SUCH_WH',
                                   database='DB', schema_name='PUBLIC')
    with pytest.raises(ValueError) as error:
        connector.session()
    assert str(error.value) == (
        "Connector warehouse 'KUMO_SHEET_CHECK_NO_SUCH_WH' differs from the "
        "shared Snowpark session warehouse 'KUMO_PYTORCH_USABILITY_WH'. "
        'Switch the session warehouse, or supply an identity for a separate '
        'session. Use SnowflakeConnector.from_session() to take the '
        "session's warehouse and role.")


@pytest.mark.parametrize('field', ['database', 'schema'])
def test_sql_source_names_the_missing_context(notebook_session, monkeypatch,
                                              field):
    for name in ('database', 'schema'):
        monkeypatch.setattr(notebook_session, f'get_current_{name}',
                            lambda: '"KUMO_PYTORCH_USABILITY"')
    connector = SnowflakeConnector.from_session(notebook_session)
    monkeypatch.setattr(notebook_session, f'get_current_{field}', lambda: None)
    with pytest.raises(ValueError) as error:
        _ = connector.sql('SELECT 1 AS value').columns
    assert str(error.value) == (
        f'The shared Snowpark session has no current {field}; SQL sources '
        f"require the connector {field} 'KUMO_PYTORCH_USABILITY'. Call "
        'session.use_database(...) and session.use_schema(...), or use '
        'SnowflakeConnector.from_session() without database or schema '
        'overrides.')


def test_from_session_rejects_a_session_owned_by_an_identity_connector(
        notebook_session, monkeypatch):
    from kumo_snowflake import session as sessions
    from snowflake.snowpark import Session

    from kumo.engine.connector.identity import SnowflakeIdentity

    owned = Session.builder.config('local_testing', True).create()
    monkeypatch.setattr(sessions, '_sessions', sessions._SessionCache())
    monkeypatch.setattr(Session.SessionBuilder, 'create', lambda _: owned)
    identity = SnowflakeIdentity(account='owned-session-test',
                                 username='test', password='test')
    identity_connector = SnowflakeConnector(
        warehouse='MOCK_WAREHOUSE', database='MOCK_DATABASE',
        schema_name='MOCK_SCHEMA', identity=identity)
    try:
        assert identity_connector.session() is owned
        with pytest.raises(ValueError, match='session is owned by Kumo'):
            SnowflakeConnector.from_session(owned)
        external = SnowflakeConnector.from_session(notebook_session)
        assert external.session() is notebook_session
    finally:
        sessions.close(identity.slug)
        owned.close()
