# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The live tier: the reader against a real Snowflake table, row-verified."""

from __future__ import annotations

import json
import os

import pytest
from kumo_core.sources.errors import PermanentSourceError
from kumo_core.sources.types import ReadMode
from kumo_snowflake.identity import hour_version
from kumoapi.typing import Dtype
from sources.conformance import SourceReaderConformance

CREDENTIALS = os.environ.get("KUMO_SNOWFLAKE_CREDS",
                             "/home/vedaant/snowflake-creds/creds.json")
ROLE, WAREHOUSE = "SDK_MINIMAL", "KUMO_XS"
DATABASE, SCHEMA, TABLE = "KUMO_TESTS", "HM_SAMPLED", "TRANSACTIONS"
VIEW, MATERIALIZED_VIEW = "TRANS_VIEW", "TRANS_VIEW_MAT"
KNOWN_ROWS = 1586527
COLUMNS = {
    "T_DAT": Dtype.time,
    "CUSTOMER_ID": Dtype.string,
    "ARTICLE_ID": Dtype.int64,
    "PRICE": Dtype.float,
    "SALES_CHANNEL_ID": Dtype.int64,
}

if not os.path.exists(CREDENTIALS):
    pytest.skip(
        f"LIVE TIER NOT RUN: no Snowflake credentials at {CREDENTIALS}. "
        f"Nothing in this file was covered. Point KUMO_SNOWFLAKE_CREDS at a "
        f"JSON file holding account, user and password for a role with USAGE "
        f"on {DATABASE}.{SCHEMA} and {WAREHOUSE} plus SELECT on {TABLE}.",
        allow_module_level=True,
    )

_STATE: dict = {}


def connection():
    if "connection" not in _STATE:
        import snowflake.connector
        with open(CREDENTIALS, encoding="utf-8") as handle:
            secrets = json.load(handle)
        _STATE["connection"] = snowflake.connector.connect(
            account=secrets["account"],
            user=secrets["user"],
            password=secrets["password"],
            role=ROLE,
            warehouse=WAREHOUSE,
            database=DATABASE,
            schema=SCHEMA,
        )
    return _STATE["connection"]


def live_reader():
    if "reader" not in _STATE:
        from kumo_snowflake import SnowflakeSourceReader
        _STATE["reader"] = SnowflakeSourceReader(connection(),
                                                 database=DATABASE,
                                                 schema=SCHEMA)
    return _STATE["reader"]


def query(sql: str):
    cursor = connection().cursor()
    try:
        cursor.execute(sql)
        return cursor.fetchall()
    finally:
        cursor.close()


def counted_rows() -> int:
    """Row count established by the warehouse, not by the reader."""
    if "rows" not in _STATE:
        _STATE["rows"] = query(
            f'SELECT COUNT(*) FROM "{DATABASE}"."{SCHEMA}"."{TABLE}"')[0][0]
    return _STATE["rows"]


@pytest.fixture(scope="module", autouse=True)
def close_the_connection():
    yield
    opened = _STATE.pop("connection", None)
    if opened is not None:
        opened.close()


class TestSnowflakeConformance(SourceReaderConformance):
    """The shared contract, unmodified, against a minimal-privilege role."""
    def reader(self):
        return live_reader()

    def table(self) -> str:
        return TABLE

    def columns(self):
        return COLUMNS

    def truth_rows(self) -> int:
        return counted_rows()


class TestSnowflakeQueryConformance(SourceReaderConformance):
    """The same contract against a query rather than a table."""
    def reader(self):
        return live_reader()

    def table(self) -> str:
        return "custom_sql_live_conformance"

    def sql(self) -> str:
        projection = ", ".join(f'"{name}"' for name in COLUMNS)
        return (f'SELECT {projection} FROM '
                f'"{DATABASE}"."{SCHEMA}"."{TABLE}"')

    def columns(self):
        return COLUMNS

    def truth_rows(self) -> int:
        return counted_rows()


def test_the_fixture_is_the_table_and_the_views_this_tier_claims():
    """Two views of one kind would leave the materialized branch uncovered."""
    assert counted_rows() == KNOWN_ROWS
    assert dict(
        query(f"SELECT TABLE_NAME, TABLE_TYPE FROM "
              f'"{DATABASE}".INFORMATION_SCHEMA.TABLES WHERE '
              f"TABLE_SCHEMA = '{SCHEMA}' AND TABLE_NAME IN "
              f"('{VIEW}', '{MATERIALIZED_VIEW}')")) == {
                  VIEW: "VIEW",
                  MATERIALIZED_VIEW: "MATERIALIZED VIEW"
              }


def test_a_minimal_privilege_role_can_read_without_new_grants():
    assert query("SELECT CURRENT_ROLE()")[0][0] == ROLE


def test_the_stage_is_left_empty_after_a_read(tmp_path):
    reader = live_reader()
    reader.prepare(TABLE, str(tmp_path / "cleanup"), columns=COLUMNS,
                   mode=ReadMode.STAGED)
    assert query(f"LIST '@{reader._stage}/{TABLE}/'") == []


def test_a_physical_table_is_dated_by_the_second_it_was_last_altered():
    version = live_reader().data_version(TABLE)
    assert version.isdigit()
    assert version != hour_version()


@pytest.mark.parametrize("view", [VIEW, MATERIALIZED_VIEW])
def test_a_view_of_either_kind_falls_back_to_the_hour_window(view):
    assert live_reader().data_version(view) == hour_version()


def test_an_absent_source_is_reported_not_quietly_dated():
    with pytest.raises(PermanentSourceError):
        live_reader().data_version("KUMO_CONNECTOR_NO_SUCH_TABLE")
