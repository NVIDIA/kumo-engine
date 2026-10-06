# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stamping a Delta table from its transaction log, and what has no log."""

import pytest
from databricks.sql.exc import Error, ServerOperationError
from fakes import FakeCursor
from kumo_core.sources.errors import SourceNotFoundError
from kumo_databricks.identity import column_types, data_version, hour_version

SOURCE = "`samples`.`tpch`.`customer`"
COLUMN_ROWS = [
    ("c_custkey", "bigint", None),
    ("c_name", "string", None),
    ("c_acctbal", "decimal(18,2)", None),
    ("", "", ""),
    ("# Detailed Table Information", "", ""),
    ("Catalog", "samples", ""),
    ("Type", "MANAGED", ""),
]
PARTITIONED_ROWS = [
    ("c_custkey", "bigint", None),
    ("Type", "date", None),
    ("# Partition Information", "", ""),
    ("# col_name", "data_type", "comment"),
    ("Type", "date", None),
    ("", "", ""),
    ("# Detailed Table Information", "", ""),
    ("Type", "MANAGED", ""),
]
COMMITTED = "2026-08-27T15:07:20"
HISTORY_ROWS = [(947, COMMITTED, "CREATE OR REPLACE TABLE AS SELECT")]
COMMIT_VERSION = "947_1787843240"


def cursor(columns=None, history=None, raises=None):
    return FakeCursor(
        {
            "DESCRIBE EXTENDED": COLUMN_ROWS if columns is None else columns,
            "DESCRIBE HISTORY": HISTORY_ROWS if history is None else history,
        }, raises=raises)


def stamp(recorder=None, source=SOURCE, **overrides):
    return data_version(recorder or cursor(**overrides), source)


def test_a_delta_table_is_stamped_with_its_version_and_commit_time():
    """The first two history columns are the stamp; a fraction is dropped."""
    assert stamp() == COMMIT_VERSION
    assert stamp(history=[(947, f"{COMMITTED}.9", "WRITE",
                           "extra")]) == (COMMIT_VERSION)


def test_the_stamp_changes_when_the_same_version_is_recommitted():
    assert stamp() != stamp(history=[(947, "2026-08-28T15:07:20", "WRITE")])
    assert stamp() != stamp(history=[(948, COMMITTED, "WRITE")])


def test_a_table_with_no_history_falls_back_to_the_hour_window():
    assert stamp(history=[]) == hour_version()


@pytest.mark.parametrize("refusal", [
    ServerOperationError("[EXPECT_TABLE_NOT_VIEW] not a table"),
    ServerOperationError("[TABLE_OR_VIEW_NOT_FOUND_ELSEWHERE] nope"),
    Error("connection reset"),
])
def test_a_history_that_cannot_be_read_falls_back_to_the_hour_window(refusal):
    """A view has no Delta log, and its history is refused rather than empty."""
    assert stamp(cursor(raises={"DESCRIBE HISTORY": refusal})) == hour_version()


@pytest.mark.parametrize("refusal", [
    ServerOperationError("[TABLE_OR_VIEW_NOT_FOUND] no such table"),
    ServerOperationError("[TABLE_OR_VIEW_NOT_FOUND.WITH_SUGGESTION] orders?"),
    ServerOperationError("[TABLE_OR_VIEW_NOT_FOUND] The table or view "
                         "`main`.`db`.`orders` cannot be found."),
    ServerOperationError("something else",
                         {"error_code": "TABLE_OR_VIEW_NOT_FOUND"}),
])
def test_a_missing_table_is_reported_not_quietly_dated(refusal):
    with pytest.raises(SourceNotFoundError):
        stamp(cursor(raises={"DESCRIBE HISTORY": refusal}))


def test_a_sentence_without_the_code_is_not_read_as_a_missing_table():
    refusal = ServerOperationError("The table or view `orders` cannot be "
                                   "found.")
    assert stamp(cursor(raises={"DESCRIBE HISTORY": refusal})) == hour_version()


def test_the_stamp_reads_one_bounded_history_and_probes_nothing_else():
    """The history itself refuses a view, so a second statement buys nothing."""
    recorder = cursor()
    stamp(recorder)
    assert recorder.sql() == [f"DESCRIBE HISTORY {SOURCE} LIMIT 1"]


def test_a_description_reports_the_source_type_of_every_column():
    assert column_types(cursor(), SOURCE) == {
        "c_custkey": "bigint",
        "c_name": "string",
        "c_acctbal": "decimal(18,2)",
    }


def test_a_partitioned_table_never_resumes_after_its_first_heading():
    assert column_types(cursor(columns=PARTITIONED_ROWS), SOURCE) == {
        "c_custkey": "bigint",
        "Type": "date",
    }


def test_description_preserves_quoted_names_that_resemble_section_markers():
    assert column_types(
        cursor(columns=[
            (' amount ', 'decimal(38,18)', None),
            ('#tag', 'string', None),
            ('# Partition Information', 'bigint', None),
            ('   ', 'string', None),
            ('# Partition Information', '', None),
            ('amount', 'decimal(38,18)', None),
        ]), SOURCE) == {
            ' amount ': 'decimal(38,18)',
            '#tag': 'string',
            '# Partition Information': 'bigint',
            '   ': 'string',
        }
