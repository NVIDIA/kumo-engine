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
