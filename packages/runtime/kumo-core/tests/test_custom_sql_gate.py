# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The namespace gate, against text that is not what the warehouse runs.

Whatever suffix a query carries, the text ruled on is the text
:func:`subquery_safe_sql` produces.
"""

from __future__ import annotations

import pytest
from kumo_core.custom_sql import (
    advisory_read_only_check,
    custom_sql_synthetic_name,
    referenced_tables_or_none,
    subquery_safe_sql,
    validate_referenced_table_scope,
    validate_sql_output_contract,
)
from kumo_core.types.enums import ConnectorType

DIALECTS = [
    ConnectorType.SNOWFLAKE, ConnectorType.DATABRICKS, ConnectorType.BIGQUERY
]
NAMESPACE = ("db_prd", "util")
OUTSIDE = "SELECT a FROM other_db.other_schema.secrets"
#: Each parses as an ``exp.Block`` rather than a ``Select``.
SUFFIXES = [";;", ";\n-- trailing", "; /*trailing*/", " ; ; "]


@pytest.mark.parametrize("connector_type", DIALECTS)
@pytest.mark.parametrize("suffix", SUFFIXES)
def test_a_trailing_terminator_cannot_hide_an_out_of_namespace_read(
        connector_type, suffix):
    assert referenced_tables_or_none(OUTSIDE + suffix, connector_type)
    with pytest.raises(ValueError, match="outside"):
        validate_referenced_table_scope(OUTSIDE + suffix, connector_type,
                                        NAMESPACE)


@pytest.mark.parametrize("connector_type", DIALECTS)
@pytest.mark.parametrize("suffix", SUFFIXES)
def test_the_text_ruled_on_is_the_text_that_would_run(connector_type, suffix):
    executed = subquery_safe_sql(OUTSIDE + suffix, connector_type)
    assert referenced_tables_or_none(
        OUTSIDE + suffix,
        connector_type) == referenced_tables_or_none(executed, connector_type)


@pytest.mark.parametrize("connector_type", DIALECTS)
@pytest.mark.parametrize("sql", [
    "SELECT a FROM orders",
    "SELECT a FROM orders;",
    "SELECT a FROM orders;;",
    "WITH c AS (SELECT a FROM orders) SELECT a FROM c;;",
    "SELECT a FROM orders UNION ALL SELECT a FROM orders;;",
    "(SELECT a FROM orders)",
])
def test_an_in_namespace_query_still_passes_however_it_is_terminated(
        connector_type, sql):
    advisory_read_only_check(sql, connector_type)
    validate_referenced_table_scope(sql, connector_type, NAMESPACE)


def test_two_statements_are_still_two_statements():
    """Stripping the *trailing* terminator must not swallow an interior one."""
    with pytest.raises(ValueError, match="single SQL statement"):
        advisory_read_only_check("SELECT a FROM t; SELECT b FROM t",
                                 ConnectorType.SNOWFLAKE)


@pytest.mark.parametrize("sql,message", [
    ("SELECT a + 1 FROM orders;;", "explicit alias"),
    ("SELECT AS STRUCT a FROM orders;;", "flat named columns"),
])
def test_the_output_contract_is_not_skipped_by_a_trailing_terminator(
        sql, message):
    with pytest.raises(ValueError, match=message):
        validate_sql_output_contract(sql, ConnectorType.BIGQUERY)


@pytest.mark.parametrize("connector_type", DIALECTS)
def test_identity_is_unmoved_by_the_fix(connector_type):
    assert custom_sql_synthetic_name(
        OUTSIDE,
        connector_type) == (custom_sql_synthetic_name(OUTSIDE + ";;",
                                                      connector_type))


def test_a_cost_the_warehouse_declined_to_give_is_over_budget() -> None:
    from kumo_core.custom_sql import (
        SQL_SAMPLE_MAX_SCAN_BYTES,
        sample_over_budget,
    )

    assert sample_over_budget(None, gated=True)
    assert sample_over_budget(SQL_SAMPLE_MAX_SCAN_BYTES + 1, gated=True)
    assert not sample_over_budget(SQL_SAMPLE_MAX_SCAN_BYTES, gated=True)
    assert not sample_over_budget(0, gated=True)

    # Ungated, `None` means there is nothing to gate.
    assert not sample_over_budget(None, gated=False)
    assert not sample_over_budget(SQL_SAMPLE_MAX_SCAN_BYTES + 1, gated=False)


@pytest.mark.parametrize('count', [-1, '1; DROP TABLE events'])
def test_sample_select_rejects_invalid_row_limit(count):
    from kumo_core.custom_sql import sample_select

    with pytest.raises(ValueError):
        sample_select('(SELECT 1)', count)


def test_sample_select_zero_is_bounded():
    from kumo_core.custom_sql import sample_select

    assert sample_select('(SELECT 1)', 0) == 'SELECT * FROM (SELECT 1) LIMIT 0'
