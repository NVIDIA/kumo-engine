# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from snowflake.connector.constants import FIELD_NAME_TO_ID

import io_utils.snowflake.session
from io_utils.snowflake.query import SnowflakeField, SnowflakeQuery
from io_utils.snowflake.session import get_fsutil_session
from kumo.testing import withSnowflakeCredentials
from kumo.testing.snowflake import setup_full_snowflake_session


def test_validation_error_omits_provider_row_value(caplog):
    class Response(BaseModel):
        a: str = SnowflakeField.text()

    canary = 'customer-row-value password=query-secret-canary'
    cursor = MagicMock()
    cursor.description = [
        SimpleNamespace(name='a', type_code=FIELD_NAME_TO_ID['TEXT'])
    ]
    cursor.fetchall.return_value = [([canary],)]

    with pytest.raises(ValueError) as exc_info:
        SnowflakeQuery._get_result(cursor, Response)

    assert "('a', 'list')" in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert canary not in str(exc_info.value)
    assert canary not in repr(exc_info.value)
    assert canary not in caplog.text


@pytest.fixture
@withSnowflakeCredentials
def patched_session(monkeypatch, tempdir):
    session = setup_full_snowflake_session(tempdir)
    monkeypatch.setattr(io_utils.snowflake.session, '_fsutil_session', session)
    yield session


def test_simple_schema(patched_session):
    class Response(BaseModel):
        a: str = SnowflakeField.text()

    with get_fsutil_session().connection.cursor() as cur:
        content = 'some_text'
        cur.execute(f"SELECT '{content}' AS \"a\"")
        res = SnowflakeQuery._get_result(cur, Response)
        assert len(res) == 1
        assert res[0].a == content


def test_complex_schema(patched_session):
    class Response(BaseModel):
        A: str = SnowflakeField.text()
        B: int = SnowflakeField.fixed()
        C: bool = SnowflakeField.boolean()
        D: datetime = SnowflakeField.timestamp_ntz()

    with get_fsutil_session().connection.cursor() as cur:
        cur.execute("""SELECT
                    'text' AS A,
                    12345 AS B,
                    True AS C,
                    SYSDATE() AS D
                    """)
        res = SnowflakeQuery._get_result(cur, Response)
        assert len(res) == 1
        assert res[0].A == 'text'
        assert res[0].B == 12345
        assert res[0].C is True
        assert (datetime.now() - res[0].D).total_seconds() < 60
