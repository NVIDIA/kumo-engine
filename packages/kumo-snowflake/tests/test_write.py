# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Snowflake artifact-table imports."""

import traceback
from types import SimpleNamespace

import pytest
from kumo_snowflake.errors import is_lapsed_credential
from kumo_snowflake.write import (
    ArtifactWriteError,
    export_table_files,
    overwrite_table,
    write_artifacts_to_table,
)
from snowflake.connector.errors import ProgrammingError


class _Connection:
    """Minimal shared-writer connection that records issued statements."""

    context = object()

    def __init__(self) -> None:
        self.statements: list[str] = []

    def _qualified_table_name(self, table_name: str) -> str:
        return f'"DB"."PUBLIC"."{table_name}"'

    def table_exists(self, table_name: str) -> bool:
        del table_name
        return False

    def execute_meta_query(self, context, statement: str, *, creds_clause=None):
        del context, creds_clause
        self.statements.append(statement)

    def write_artifacts_to_table(self, *args, **kwargs):
        write_artifacts_to_table(self, *args, **kwargs)

    def drop_table_if_exists(self, table_name):
        self.execute_meta_query(
            self.context,
            f'DROP TABLE IF EXISTS {self._qualified_table_name(table_name)}')


def test_streamed_parquet_export_preserves_timestamp_logical_types() -> None:
    connection = _Connection()

    write_artifacts_to_table(connection, "PREDICTIONS", "s3://bucket/run/",
                             replace=True)

    create_format = next(
        statement for statement in connection.statements
        if statement.startswith("CREATE TEMPORARY FILE FORMAT"))
    copy = next(statement for statement in connection.statements
                if statement.startswith("COPY INTO"))
    assert "TYPE = PARQUET USE_LOGICAL_TYPE = TRUE" in create_format
    assert "FILE_FORMAT = (TYPE = PARQUET USE_LOGICAL_TYPE = TRUE)" in copy


def test_overwrite_maps_staging_columns_by_destination_name() -> None:
    connection = _Connection()
    connection.table_exists = lambda _name: True
    connection.table_columns = lambda name: (["SCORE", "ENTITY", "TIMESTAMP"]
                                             if name == "PREDICTIONS" else
                                             ["TIMESTAMP", "SCORE", "ENTITY"])

    overwrite_table(connection, "STAGING", "PREDICTIONS",
                    quote=connection._qualified_table_name)

    assert connection.statements == [
        'INSERT OVERWRITE INTO "DB"."PUBLIC"."PREDICTIONS" '
        '("SCORE", "ENTITY", "TIMESTAMP") '
        'SELECT "SCORE", "ENTITY", "TIMESTAMP" '
        'FROM "DB"."PUBLIC"."STAGING"'
    ]


def test_overwrite_rejects_a_different_staging_schema() -> None:
    connection = _Connection()
    connection.table_exists = lambda _name: True
    connection.table_columns = lambda name: (["SCORE", "ENTITY"] if name ==
                                             "PREDICTIONS" else ["SCORE"])

    with pytest.raises(ArtifactWriteError, match="different columns"):
        overwrite_table(connection, "STAGING", "PREDICTIONS",
                        quote=connection._qualified_table_name)


@pytest.mark.parametrize('role_source', ['tracked', 'query', 'unavailable'])
@pytest.mark.parametrize('replace', [False, True])
def test_artifact_write_reports_safe_diagnostics(role_source, replace,
                                                 caplog) -> None:
    connection = _Connection()
    sql = "COPY INTO PRIVATE_TABLE CREDENTIALS = "
    aws_key = 'fake-aws-access-key'
    failure = ProgrammingError(msg=f"{sql}(AWS_KEY_ID = '{aws_key}')",
                               errno=3001, sqlstate='42501',
                               sfqid='01b12345-0000-0000-0000-000000000001')
    role_queries = []
    closed = []

    class Cursor:
        def execute(self, statement):
            role_queries.append(statement)
            if role_source == 'unavailable':
                raise RuntimeError('Role lookup failed')

        def fetchone(self):
            return ('OUTPUT_WRITER',)

        def close(self):
            closed.append(True)

    connection.context = SimpleNamespace(
        role='OUTPUT_WRITER' if role_source == 'tracked' else None,
        cursor=Cursor)

    def execute(context, statement, *, creds_clause=None):
        connection.statements.append(statement)
        if statement.startswith('COPY INTO'):
            raise failure

    connection.execute_meta_query = execute

    with pytest.raises(ArtifactWriteError) as error:
        export_table_files(connection, 'PREDICTIONS', 's3://bucket/run/',
                           replace=replace)

    message = str(error.value)
    assert '"DB"."PUBLIC"."PREDICTIONS"' in message
    assert '_STAGING_' not in message
    copy = next(statement for statement in connection.statements
                if statement.startswith('COPY INTO'))
    assert ('_STAGING_' in copy) == replace
    assert 'ProgrammingError errno=3001 sqlstate=42501' in message
    assert 'query=01b12345-0000-0000-0000-000000000001' in message
    assert 'the role lacks a privilege this write needs' in message
    if role_source == 'unavailable':
        assert 'as role' not in message
    else:
        assert 'as role OUTPUT_WRITER' in message
    assert role_queries == ([] if role_source == 'tracked' else
                            ['SELECT CURRENT_ROLE()'])
    assert closed == ([] if role_source == 'tracked' else [True])
    assert error.value.errno == 3001
    assert error.value.sqlstate == '42501'
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
    assert message in caplog.messages
    rendered = ''.join(traceback.format_exception(error.value))
    for output in (message, caplog.text, rendered):
        assert sql not in output
        assert aws_key not in output

    with pytest.raises(ArtifactWriteError) as next_error:
        write_artifacts_to_table(connection, 'NEXT', 's3://bucket/run/')
    assert 'Failed to write "DB"."PUBLIC"."NEXT"' in str(next_error.value)


class _ExpiredTokenError(Exception):
    errno = 390114
    sqlstate = "08001"


def test_artifact_write_preserves_expired_credential_classification() -> None:
    def upload(*_args, **_kwargs) -> None:
        raise _ExpiredTokenError()

    with pytest.raises(ArtifactWriteError) as error:
        write_artifacts_to_table(_Connection(), "PREDICTIONS",
                                 "/tmp/predictions.parquet", replace=True,
                                 upload_local=upload)

    assert is_lapsed_credential(error.value)
    assert error.value.errno == 390114
    assert error.value.sqlstate == '08001'
    message = str(error.value)
    assert '"DB"."PUBLIC"."PREDICTIONS"' in message
    assert '_ExpiredTokenError errno=390114 sqlstate=08001' in message
    assert 'as role' not in message
    assert 'privilege' not in message
