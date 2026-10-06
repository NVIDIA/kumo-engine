# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from kumo_snowflake.config import connection_toml_entry


def test_connection_toml_entry_uses_snowflake_config_manager(monkeypatch):
    manager = SimpleNamespace(
        conf_file_cache={
            'default_connection_name': 'development',
            'connections': {
                'development': {'account': 'xy12345', 'user': 'alice'},
            },
        },
        read_config=lambda: None,
    )
    monkeypatch.setattr(
        'snowflake.connector.config_manager.CONFIG_MANAGER', manager)

    assert connection_toml_entry() == {
        'account': 'xy12345',
        'user': 'alice',
    }


def test_connection_toml_entry_reports_available_names(monkeypatch):
    manager = SimpleNamespace(
        conf_file_cache={
            'connections': {
                'development': {'account': 'xy12345'},
                'production': {'account': 'example_account'},
            },
        },
        read_config=lambda: None,
    )
    monkeypatch.setattr(
        'snowflake.connector.config_manager.CONFIG_MANAGER', manager)

    with pytest.raises(
            ValueError,
            match=("has no connection 'missing'. It defines: "
                   "'development', 'production'")):
        connection_toml_entry('missing')
