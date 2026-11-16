# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A query source owes the same contract a physical table does."""

from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.errors import (
    RowCountMismatchError,
    UnsupportedReadModeError,
)
from kumo_core.sources.identity import sql_data_version
from kumo_core.sources.types import ReadMode, StagedTable
from kumo_core.types.enums import ConnectorType
from tests.sources.conformance import SourceReaderConformance

ROWS = 40
SQL = "SELECT id, amount FROM orders WHERE amount > 0"
COLUMNS = {"id": "int64", "amount": "float32"}


def fixture_table() -> pa.Table:
    return pa.table({
        "id": pa.array(range(ROWS), pa.int64()),
        "amount": pa.array([float(i) for i in range(ROWS)], pa.float32()),
        "unused": pa.array(["x"] * ROWS, pa.string()),
    })


class QueryReader:
    @property
    def source_type(self) -> ConnectorType:
        return ConnectorType.SNOWFLAKE

    def read_modes(self, table):
        return (ReadMode.STAGED, )

    def data_version(self, table, *, sql=None, columns=None):
        assert sql is not None, table
        return sql_data_version(sql, ConnectorType.SNOWFLAKE,
                                lambda leaf: "v1", columns)

    def prepare(self, table, dest, *, columns, mode, aliases=None,
                expected_rows=None, worker_ceiling=None, sql=None, version=None):
        assert sql is not None, table
        if mode is not ReadMode.STAGED:
            raise UnsupportedReadModeError(f"{mode.value} unsupported",
                                           table=table)
        os.makedirs(dest, exist_ok=True)
        source = fixture_table().select(list(columns))
        if aliases:
            source = source.rename_columns(
                [aliases.get(name, name) for name in source.column_names])
        if expected_rows is not None and source.num_rows != expected_rows:
            raise RowCountMismatchError(expected=expected_rows,
                                        actual=source.num_rows, table=table)
        path = f"{dest}/part-00000.parquet"
        pq.write_table(source, path)
        return StagedTable(directory=dest, paths=(path, ),
                           version=version or self.data_version(
                               table, sql=sql, columns=columns),
                           source_size=os.path.getsize(path), mode_used=mode,
                           num_rows=source.num_rows)


class TestAQuerySourcePassesTheContract(SourceReaderConformance):
    def reader(self):
        return QueryReader()

    def table(self) -> str:
        return "custom_sql_0123456789abcdef"

    def sql(self) -> str:
        return SQL

    def columns(self):
        return COLUMNS

    def truth_rows(self) -> int:
        return ROWS


def test_a_reader_that_lost_the_query_would_read_the_wrong_thing(tmp_path):
    with pytest.raises(AssertionError):
        QueryReader().prepare("custom_sql_0123456789abcdef", str(tmp_path),
                              columns=COLUMNS, mode=ReadMode.STAGED)


def test_the_output_contract_is_part_of_a_querys_version():
    reader = QueryReader()
    name = "custom_sql_0123456789abcdef"
    assert reader.data_version(name, sql=SQL, columns={"id": "int64"}) != (
        reader.data_version(name, sql=SQL, columns={"id": "int32"}))
