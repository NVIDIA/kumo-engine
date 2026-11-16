# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A query's change-detection version, which no catalog can supply."""

from __future__ import annotations

import logging

import pytest
from kumo_core.custom_sql import custom_sql_synthetic_name
from kumo_core.sources import identity
from kumo_core.sources.identity import sql_data_version
from kumo_core.types.enums import ConnectorType

SQL = "SELECT id, amount FROM orders WHERE amount > 0"


#: Pin the 24-hour staleness bucket, which would otherwise roll mid-test and
#: move a version for a reason no test here is about.
@pytest.fixture(autouse=True)
def _fixed_staleness_window(monkeypatch):
    monkeypatch.setattr(identity, "time", lambda: 1_788_000_000.0)


def stamped(sql=SQL, version="v1", cols=None,
            connector=ConnectorType.SNOWFLAKE):
    return sql_data_version(sql, connector, lambda leaf: version, cols)


def test_the_same_inputs_give_the_same_version():
    assert stamped() == stamped()


def test_a_referenced_table_that_moved_gives_a_new_version():
    assert stamped(version="v1") != stamped(version="v2")


def test_the_version_keeps_the_dependency_versions_it_folds():
    version = stamped(version="v1")
    assert version.kind == "hour_bucket"
    assert set(version.dependencies.values()) == {"v1"}


def test_a_formatting_only_edit_does_not_move_the_version():
    assert stamped(
        sql="select  ID,\n AMOUNT from ORDERS where AMOUNT > 0") == (stamped())


def test_a_declared_dtype_change_moves_the_version():
    assert stamped(cols={"amount": "float32"}) != stamped(
        cols={"amount": "float64"})


def test_a_reference_that_cannot_be_stamped_never_repeats():
    def refuse(leaf):
        raise RuntimeError(leaf)

    first = sql_data_version(SQL, ConnectorType.SNOWFLAKE, refuse)
    assert first != sql_data_version(SQL, ConnectorType.SNOWFLAKE, refuse)


def test_a_query_that_no_longer_parses_never_repeats():
    unparseable = "this is not sql at all"
    assert sql_data_version(unparseable, ConnectorType.SNOWFLAKE,
                            lambda leaf: "v1") != sql_data_version(
                                unparseable, ConnectorType.SNOWFLAKE,
                                lambda leaf: "v1")


def test_the_synthetic_name_is_the_identity_the_platform_derives():
    assert custom_sql_synthetic_name(
        SQL, ConnectorType.SNOWFLAKE) == (custom_sql_synthetic_name(
            "select  ID,\n AMOUNT from ORDERS "
            "where AMOUNT > 0", ConnectorType.SNOWFLAKE))


def test_a_probe_that_answers_nothing_never_repeats():
    first = sql_data_version(SQL, ConnectorType.SNOWFLAKE, lambda leaf: None)
    assert first != sql_data_version(SQL, ConnectorType.SNOWFLAKE,
                                     lambda leaf: None)


def test_two_leaves_nobody_could_stamp_do_not_share_one_version():
    joined = "SELECT o.id FROM orders o JOIN shipments s ON s.id = o.id"
    unknown = sql_data_version(joined, ConnectorType.SNOWFLAKE,
                               lambda leaf: None)
    assert unknown != sql_data_version(joined, ConnectorType.SNOWFLAKE,
                                       lambda leaf: "None")


def test_a_reference_nobody_could_resolve_is_named_in_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=identity.__name__):
        identity.sql_data_version(SQL, ConnectorType.SNOWFLAKE,
                                  lambda leaf: None, table="orders_recent")
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "orders_recent" in message
    assert "ORDERS" in message, 'the leaf that could not be resolved'
    assert "re-materializing conservatively" in message


def test_a_probe_that_raised_says_what_it_raised(caplog):
    with caplog.at_level(logging.WARNING, logger=identity.__name__):
        identity.sql_data_version(
            SQL, ConnectorType.SNOWFLAKE, lambda leaf:
            (_ for _ in ()).throw(RuntimeError("no grant")),
            table="orders_recent")
    assert "no grant" in caplog.records[0].getMessage()


def test_a_query_that_no_longer_parses_says_so(caplog):
    with caplog.at_level(logging.WARNING, logger=identity.__name__):
        identity.sql_data_version("this is not sql at all",
                                  ConnectorType.SNOWFLAKE, lambda leaf: "v1",
                                  table="orders_recent")
    message = caplog.records[0].getMessage()
    assert "orders_recent" in message
    assert "could not be parsed" in message
    assert "re-materializing conservatively" in message
