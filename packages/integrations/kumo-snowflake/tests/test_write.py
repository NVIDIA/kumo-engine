# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Snowflake artifact-table imports."""

import pytest
from kumo_snowflake.errors import is_lapsed_credential
from kumo_snowflake.write import (
    ArtifactWriteError,
    overwrite_table,
    write_artifacts_to_table,
)


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
