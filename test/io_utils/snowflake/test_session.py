# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import io_utils.snowflake.session
from io_utils.snowflake.session import snowpark_session_in_app_schema
from kumo.testing.snowflake import setup_full_snowflake_session


@pytest.mark.private_resource('snowflake')
def test_schema_no_change(monkeypatch, tempdir):
    session = setup_full_snowflake_session(tempdir)
    original_database = session.get_current_database()
    original_schema = session.get_current_schema()

    # no change
    monkeypatch.setattr(
        io_utils.snowflake.session,
        'SNOWFLAKE_DATABASE',
        None,
    )
    monkeypatch.setattr(
        io_utils.snowflake.session,
        'SNOWFLAKE_APP_SCHEMA',
        None,
    )

    with snowpark_session_in_app_schema(session):
        assert session.get_current_database() == original_database
        assert session.get_current_schema() == original_schema

    assert session.get_current_database() == original_database
    assert session.get_current_schema() == original_schema


@pytest.mark.private_resource('snowflake')
def test_schema_change(monkeypatch, tempdir):
    session = setup_full_snowflake_session(tempdir)
    original_database = session.get_current_database()
    original_schema = session.get_current_schema()
    new_database = '"KUMO_APP_DB"'
    new_schema = '"KUMO_SCHEMA"'

    monkeypatch.setattr(
        io_utils.snowflake.session,
        'SNOWFLAKE_DATABASE',
        new_database,
    )
    monkeypatch.setattr(
        io_utils.snowflake.session,
        'SNOWFLAKE_APP_SCHEMA',
        new_schema,
    )

    with snowpark_session_in_app_schema(session):
        assert session.get_current_database() == new_database
        assert session.get_current_schema() == new_schema

    assert session.get_current_database() == original_database
    assert session.get_current_schema() == original_schema
