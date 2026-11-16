# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Reading a Snowflake ``connections.toml`` entry into identity arguments.

An identity holds credentials, so a named connection has to be read into one.
These cover the reading, not the connector binding, which needs a backend.
"""
import pytest
from kumoapi.data_source import (
    SnowflakeAuthenticationKind,
    SnowflakeIdentityConfig,
)

from kumoai_studio.connector import snowflake_connector as module


def _config(args) -> SnowflakeIdentityConfig:
    r"""Narrows what ``CreateIdentityArgs`` holds to the Snowflake member.

    ``config`` is the union of every provider, so reading an account off one
    is meaningful only once the kind is established.
    """
    config = args.config
    assert isinstance(config, SnowflakeIdentityConfig)
    return config


@pytest.fixture
def connections(monkeypatch):
    """Replaces what the config manager read, without touching a real file."""
    entries = {}

    class _Manager:
        conf_file_cache = {'connections': entries}

        def read_config(self):
            pass

    monkeypatch.setattr('snowflake.connector.config_manager.CONFIG_MANAGER',
                        _Manager())
    return entries


def test_a_key_pair_entry_becomes_key_pair_credentials(connections, tmp_path):
    key = tmp_path / 'rsa_key.p8'
    key.write_text('test-private-key-material')
    connections['dev'] = {
        'account': 'xy12345',
        'user': 'KUMO_READER',
        'private_key_file': str(key),
    }

    args = module._connections_toml_identity_args(
        'sf', module._connections_toml_entry('dev'))

    assert _config(args).account == 'xy12345'
    assert _config(args).username == 'KUMO_READER'
    assert (_config(args).authentication
            is SnowflakeAuthenticationKind.KEY_PAIR)
    assert 'PRIVATE KEY' not in repr(args.credentials)


def test_a_password_entry_becomes_password_credentials(connections):
    connections['dev'] = {
        'account': 'xy12345',
        'user': 'KUMO_READER',
        'password': 'hunter2',
    }

    args = module._connections_toml_identity_args(
        'sf', module._connections_toml_entry('dev'))

    assert (_config(args).authentication
            is SnowflakeAuthenticationKind.PASSWORD)
    assert 'hunter2' not in repr(args.credentials)


def test_a_key_pair_wins_over_a_password(connections, tmp_path):
    r"""What ``snowflake.connector`` itself does with the same file."""
    key = tmp_path / 'rsa_key.p8'
    key.write_text('test-private-key-material')
    connections['dev'] = {
        'account': 'xy12345',
        'user': 'U',
        'password': 'hunter2',
        'private_key_file': str(key),
    }

    args = module._connections_toml_identity_args(
        'sf', module._connections_toml_entry('dev'))

    assert (_config(args).authentication
            is SnowflakeAuthenticationKind.KEY_PAIR)


def test_an_sso_entry_says_why_it_cannot_become_an_identity(connections):
    r"""A browser or SSO connection has nothing to store, and an identity that
    holds no material cannot open a connection later.
    """
    connections['dev'] = {
        'account': 'xy12345',
        'user': 'U',
        'authenticator': 'externalbrowser',
    }

    with pytest.raises(ValueError, match='neither a password nor a private'):
        module._connections_toml_identity_args(
            'sf', module._connections_toml_entry('dev'))


def test_the_default_connection_is_taken_when_none_is_named(
        connections, monkeypatch):
    connections['dev'] = {'account': 'a', 'user': 'u', 'password': 'p'}

    class _Manager:
        conf_file_cache = {
            'connections': connections,
            'default_connection_name': 'dev',
        }

        def read_config(self):
            pass

    monkeypatch.setattr('snowflake.connector.config_manager.CONFIG_MANAGER',
                        _Manager())

    assert module._connections_toml_entry(None)['account'] == 'a'


def test_an_absent_entry_names_what_the_file_defines(connections):
    connections['dev'] = {'account': 'a', 'user': 'u', 'password': 'p'}

    with pytest.raises(ValueError, match="defines: 'dev'"):
        module._connections_toml_entry('prod')
