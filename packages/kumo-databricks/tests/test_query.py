# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from fakes import FakeFiles
from kumo_core.sources.types import ReadMode
from kumo_databricks.sql import normalize_table_coordinate, quote_identifier
from test_source import (
    COLUMNS,
    DESCRIPTION,
    HISTORY,
    StagingCursor,
    arrow,
    build,
)

SQL = "SELECT c_custkey, c_acctbal FROM customer WHERE c_acctbal > 0"
NAME = "custom_sql_0123456789abcdef"
QUERY_DESCRIPTION = [
    ("c_custkey", "bigint", None),
    ("c_acctbal", "decimal(38,2)", None),
]


@pytest.mark.parametrize(
    ("coordinate", "expected"),
    [
        ("table-name", ("sales-schema", "table-name")),
        (
            "sales-schema.订单@#$%20?;quote'back\\slash",
            ("sales-schema", "订单@#$%20?;quote'back\\slash"),
        ),
        ("main.sales-schema.tick`name", ("sales-schema", "tick`name")),
        ("MAIN.Sales-Schema.orders", ("Sales-Schema", "orders")),
    ],
)
def test_table_coordinate_normalization_is_provider_owned(coordinate, expected):
    assert normalize_table_coordinate(
        coordinate,
        catalog="main",
        schema="sales-schema",
    ) == expected


@pytest.mark.parametrize(
    ("coordinate", "message"),
    [
        ("other.sales-schema.orders", "connector catalog"),
        ("main.other.orders", "connector schema"),
        ("other.orders", "connector schema"),
        ("a.b.c.d", "must be unqualified"),
        ("", "incomplete"),
        ("sales-schema.", "incomplete"),
        ("bad name", "does not permit"),
        ("bad/name", "does not permit"),
        ("bad\x7f", "does not permit"),
        ("x" * 256, "does not permit"),
    ],
)
def test_table_coordinate_normalization_has_one_validation_rule(
        coordinate, message):
    with pytest.raises(ValueError, match=message):
        normalize_table_coordinate(
            coordinate,
            catalog="main",
            schema="sales-schema",
        )


def test_identifier_quoting_is_provider_owned():
    assert quote_identifier("tick`name") == "`tick``name`"


def warehouse(history=HISTORY, query_description=QUERY_DESCRIPTION):
    return StagingCursor(
        {
            "DESCRIBE QUERY": query_description,
            "DESCRIBE EXTENDED": DESCRIPTION,
            "DESCRIBE HISTORY": history,
        }, arrow=arrow())


def test_the_query_is_read_in_subquery_position_and_not_rewritten(tmp_path):
    cursor = warehouse()
    build(cursor).prepare(NAME, str(tmp_path), columns=COLUMNS,
                          mode=ReadMode.STREAMED, sql=SQL)
    selects = [s for s in cursor.sql() if s.startswith("SELECT")]
    assert len(selects) == 1
    assert selects[0].endswith(f"FROM ({SQL})")
    assert NAME not in selects[0]


def test_describe_query_supplies_the_width_describe_extended_cannot(tmp_path):
    cursor = warehouse()
    build(cursor).prepare(NAME, str(tmp_path), columns=COLUMNS,
                          mode=ReadMode.STREAMED, sql=SQL)
    assert cursor.matching("DESCRIBE QUERY") == [
        f"DESCRIBE QUERY SELECT * FROM ({SQL})"
    ]
    assert not cursor.matching("DESCRIBE EXTENDED")
    assert "CAST(`c_acctbal` AS DOUBLE)" in cursor.sql()[-1]


def test_a_query_is_never_routed_through_describe_history(tmp_path):
    cursor = warehouse()
    build(cursor).data_version(NAME, sql=SQL)
    assert not cursor.matching(f"({SQL})")
    assert cursor.matching("DESCRIBE HISTORY") == [
        "DESCRIBE HISTORY `samples`.`tpch`.`customer` LIMIT 1"
    ]


def test_a_query_is_stamped_by_the_tables_it_reads():
    reader = build(warehouse())
    assert reader.data_version(NAME,
                               sql=SQL) == reader.data_version(NAME, sql=SQL)
    assert reader.data_version(NAME, sql=SQL) != reader.data_version("customer")


def test_the_staged_unload_names_the_query_not_the_synthetic_table(tmp_path):
    files = FakeFiles()
    cursor = warehouse()
    build(cursor, files).prepare(NAME, str(tmp_path), columns=COLUMNS,
                                 mode=ReadMode.STAGED, sql=SQL)
    unload = cursor.matching("DIRECTORY")
    assert len(unload) == 1
    assert unload[0].endswith(f"FROM ({SQL})")
    assert f"/{NAME}/" in unload[0]


def test_query_column_types_keeps_quoted_whitespace():
    from unittest.mock import Mock

    from kumo_databricks.identity import query_column_types

    cursor = Mock()
    cursor.fetchall.return_value = [('', 'bigint'), ('  ', 'string'),
                                    (' amount ', 'decimal(30,2)')]
    assert query_column_types(cursor, '(SELECT 1)') == {
        ' amount ': 'decimal(30,2)'
    }
