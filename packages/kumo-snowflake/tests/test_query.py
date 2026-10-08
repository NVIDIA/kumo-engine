# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests
from fakes import FakeConnection, FakeSessions
from kumo_core.sources.types import ReadMode
from kumo_snowflake import SnowflakeSourceReader
from kumoapi.typing import Dtype
from test_source import BODIES, DATABASE, SCHEMA, TABLE, warehouse

SQL = 'SELECT "ENTITY_ID", "EVENT_TIME" FROM "EVENTS" WHERE "VALUE" > 0'
NAME = "custom_sql_0123456789abcdef"
COLUMNS = {"ENTITY_ID": Dtype.string, "EVENT_TIME": Dtype.time}


@pytest.fixture(autouse=True)
def http(monkeypatch, tmp_path):
    sessions = FakeSessions(BODIES)
    monkeypatch.setattr(requests, "Session", sessions)
    # The imported fake warehouse models batched presigning outside SPCS.
    monkeypatch.setattr("kumo_snowflake.staging.SPCS_TOKEN_PATH",
                        str(tmp_path / "missing-spcs-token"))
    return sessions


def reader(cursor):
    return SnowflakeSourceReader(FakeConnection(cursor), database=DATABASE,
                                 schema=SCHEMA)


def read(cursor, tmp_path, sql=SQL):
    return reader(cursor).prepare(NAME, str(tmp_path), columns=COLUMNS,
                                  mode=ReadMode.STAGED, sql=sql)


def test_the_query_is_unloaded_in_subquery_position_and_not_rewritten(tmp_path):
    cursor = warehouse()
    read(cursor, tmp_path)
    unload = cursor.matching("COPY INTO")
    assert len(unload) == 1
    assert f"FROM ({SQL})" in unload[0]
    # Named only as the stage prefix; nothing looks it up as a table.
    assert f'."{NAME}"' not in unload[0]


def test_the_query_is_never_bound_as_a_parameter(tmp_path):
    cursor = warehouse()
    read(cursor, tmp_path, f"{SQL} AND \"SEGMENT\" LIKE '%a%'")
    carried = [(sql, params) for sql, params in cursor.statements
               if "LIKE" in sql]
    assert carried
    assert all(params is None for _, params in carried)


def test_the_stage_prefix_is_the_synthetic_name_not_the_query(tmp_path):
    cursor = warehouse()
    read(cursor, tmp_path)
    assert cursor.prefixes and cursor.prefixes[0].startswith(f"{NAME}/")


def test_a_query_is_stamped_by_the_tables_it_reads(tmp_path):
    stamp = reader(warehouse())
    assert stamp.data_version(NAME,
                              sql=SQL) == stamp.data_version(NAME, sql=SQL)
    assert stamp.data_version(NAME, sql=SQL) != stamp.data_version(TABLE)
