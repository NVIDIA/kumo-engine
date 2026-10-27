# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stamping a Snowflake object from the catalog, and typing its columns."""

from datetime import datetime, timezone

import pytest
from fakes import FakeCursor
from kumo_core.sources.errors import PermanentSourceError, SourceNotFoundError
from kumo_snowflake.identity import column_types, data_version, hour_version

DATABASE, SCHEMA, TABLE = "SYNTHETIC_DB", "SYNTHETIC_SCHEMA", "EVENTS"
ALTERED = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
ALTERED_VERSION = "1704164645"


def cursor(table_type="BASE TABLE", last_altered=ALTERED, table_name=TABLE):
    return FakeCursor({
        "TABLE_TYPE": [(table_type, )],
        "LAST_ALTERED": [(table_name, last_altered)],
    })


def stamp(recorder=None, table=TABLE, **overrides):
    return data_version(recorder or cursor(**overrides), database=DATABASE,
                        schema=SCHEMA, table=table)


def test_a_table_is_stamped_with_the_second_it_was_last_altered():
    """A version carrying `.0` is a different path segment and a different key."""
    assert stamp() == ALTERED_VERSION
    assert stamp().kind == "timestamp"
    assert stamp(last_altered=ALTERED.replace(
        microsecond=900000)) == (ALTERED_VERSION)


def test_a_table_is_matched_with_exact_identifier_case():
    assert stamp(table="events", table_name="events") == (ALTERED_VERSION)


def test_a_differently_cased_table_is_not_substituted():
    assert stamp(table_name="events") == hour_version()


@pytest.mark.parametrize("table_type", ["VIEW", "MATERIALIZED VIEW"])
def test_a_view_of_either_kind_falls_back_to_the_hour_window(table_type):
    recorder = cursor(table_type=table_type)
    assert stamp(recorder) == hour_version()
    assert stamp(cursor(table_type=table_type)).kind == "hour_bucket"
    assert recorder.matching("LAST_ALTERED") == []


def test_a_table_the_catalog_does_not_list_falls_back_to_the_hour_window():
    assert stamp(FakeCursor({"TABLE_TYPE":
                             [("BASE TABLE", )]})) == (hour_version())


def test_a_source_the_catalog_does_not_hold_is_reported_not_quietly_dated():
    """A table listing with no row for it is the one answer that means the
    name is wrong, rather than that its alteration time is simply unknown."""
    with pytest.raises(SourceNotFoundError):
        stamp(FakeCursor())


def test_the_stamp_preserves_resolved_database_schema_and_table_case():
    recorder = FakeCursor({
        "TABLE_TYPE": [("BASE TABLE", )],
        "LAST_ALTERED": [("events", ALTERED)],
    })
    data_version(recorder, database="synthetic_db", schema="synthetic_schema",
                 table="events")
    assert [params for _, params in recorder.statements] == [
        ("synthetic_schema", "events"),
        ("synthetic_schema", "events"),
    ]
    assert all('"synthetic_db".INFORMATION_SCHEMA.TABLES' in statement
               for statement in recorder.sql())


def types(declared, columns):
    return column_types(FakeCursor(declared=declared), '"T"', columns)


def test_a_column_named_as_snowflake_declared_it_keeps_that_type():
    assert types({"TS": "TIMESTAMP_TZ"}, ["TS"]) == {"TS": "TIMESTAMP_TZ"}


def test_a_lower_case_column_resolves_against_the_uppercased_echo():
    """Snowflake echoes an unquoted identifier upper-cased."""
    assert types({"TS": "TIMESTAMP_TZ"}, ["ts"]) == {"ts": "TIMESTAMP_TZ"}


def test_two_declared_columns_differing_only_by_case_resolve_neither():
    """Quoted identifiers make `ts` and `TS` two columns of two types, and
    folding them would pick one of the two casts by dictionary order."""
    with pytest.raises(PermanentSourceError):
        types({"ts": "TIMESTAMP_NTZ", "TS": "TIMESTAMP_TZ"}, ["Ts"])


def test_a_column_snowflake_reported_no_type_for_is_refused_not_guessed():
    """The wrong guess shifts every timestamp in the table by the session
    offset rather than failing."""
    with pytest.raises(PermanentSourceError):
        types({"OTHER": "TEXT"}, ["TS"])


def test_no_statement_is_compiled_when_no_column_is_a_timestamp():
    recorder = FakeCursor(declared={"TS": "TIMESTAMP_TZ"})
    assert column_types(recorder, '"T"', []) == {}
    assert recorder.sql() == []


class Connection:
    """A connection that binds under one style, as ``connect()`` fixes it."""
    def __init__(self, paramstyle):
        self._paramstyle = paramstyle


def bound(paramstyle):
    """The catalog query as it would reach the warehouse."""
    recorder = cursor()
    recorder.connection = Connection(paramstyle)
    stamp(recorder)
    return " ".join(recorder.sql())


@pytest.mark.parametrize("paramstyle, mark", [("qmark", "?"),
                                              ("pyformat", "%s"),
                                              ("format", "%s"), (None, "%s")])
def test_a_query_binds_the_way_the_connection_it_runs_on_expects(
        paramstyle, mark):
    """Snowpark opens ``qmark`` and this package's own connections open
    ``pyformat``; the reader is handed the connection either way, and the
    wrong marker reaches Snowflake as literal text."""
    assert mark in bound(paramstyle)
    assert ("%s" if mark == "?" else "?") not in bound(paramstyle)


@pytest.mark.parametrize("paramstyle", ["numeric", "named"])
def test_a_style_that_numbers_its_parameters_is_refused_not_guessed(paramstyle):
    """These queries bind two values through one repeated marker, which an
    ordinal style cannot express: ``:1`` twice would match the schema against
    both columns and report an existing table as missing."""
    with pytest.raises(PermanentSourceError, match="individually"):
        bound(paramstyle)
