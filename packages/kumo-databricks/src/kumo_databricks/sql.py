# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import Mapping

from databricks.sql.client import Cursor
from kumo_core.sources import sql
from kumoapi.typing import Dtype

QUOTE = "`"
DECIMAL = re.compile(r"^decimal\s*\(\s*(\d+)", re.IGNORECASE)
WIDEST_EXACT_PRECISION = 18
DOUBLE_DTYPES = frozenset({Dtype.float, Dtype.float32, Dtype.float64})
BIGINT_DTYPES = frozenset(
    {Dtype.int, Dtype.byte, Dtype.int16, Dtype.int32, Dtype.int64})


def quote_identifier(name: str) -> str:
    return sql.quote_identifier(name, quote=QUOTE)


def qualified_name(*parts: str) -> str:
    return sql.qualified_name(*parts, quote=quote_identifier)


def normalize_table_coordinate(
    table_name: str,
    *,
    catalog: str,
    schema: str,
) -> tuple[str, str]:
    r"""Return the raw ``(schema, table)`` inside one connector target.

    A source table can carry its catalog and schema in the table field, while
    the connector already fixes both. Qualification is therefore accepted
    only when it agrees with the connector. Values stay unquoted here and are
    escaped only when a SQL statement is rendered.
    """
    coordinate = str(table_name).split('.')
    if len(coordinate) == 3:
        target_catalog, target_schema, target_table = coordinate
        if target_catalog.casefold() != catalog.casefold():
            raise ValueError(
                f"Databricks table {table_name!r} does not match the "
                "connector catalog.")
        if target_schema.casefold() != schema.casefold():
            raise ValueError(
                f"Databricks table {table_name!r} does not match the "
                "connector schema.")
    elif len(coordinate) == 2:
        target_schema, target_table = coordinate
        if target_schema.casefold() != schema.casefold():
            raise ValueError(
                f"Databricks table {table_name!r} does not match the "
                "connector schema.")
    elif len(coordinate) == 1:
        target_schema = schema
        target_table = coordinate[0]
    else:
        raise ValueError(
            f"Databricks table {table_name!r} must be unqualified, "
            "schema-qualified, or catalog-and-schema-qualified.")

    if not target_schema or not target_table:
        raise ValueError(
            f"Databricks table coordinate {table_name!r} is incomplete.")
    for identifier in (target_schema, target_table):
        if (len(identifier) > 255
                or any(char in {' ', '/'} or ord(char) < 32 or ord(char) == 127
                       for char in identifier)):
            raise ValueError(
                f"Databricks table coordinate {table_name!r} contains a "
                "character that Unity Catalog does not permit in object "
                "names.")
    return target_schema, target_table


def quote_literal(value: str) -> str:
    """One SQL string literal. A volume path and a local file both reach a
    statement this way, and doubling any quote is what keeps a name holding
    one from ending the literal early."""
    return "'" + value.replace("'", "''") + "'"


def build_projection(
    columns: Mapping[str, Dtype],
    aliases: Mapping[str, str] | None = None,
    source_types: Mapping[str, str] | None = None,
) -> str:
    """A ``DECIMAL`` wider than 18 digits does not fit a 64-bit integer, so
    Databricks yields a 128-bit decimal the graph cannot read as a number."""
    declared = source_types or {}

    def cast(name: str, dtype: Dtype, quoted: str) -> str:
        target = _cast_target(dtype, declared.get(name))
        return f"CAST({quoted} AS {target})" if target else quoted

    return sql.render_projection(columns, aliases, quote=quote_identifier,
                                 cast=cast)


def read_query(source: str, projection: str) -> str:
    return f"SELECT {projection} FROM {source}"


def _cast_target(dtype: Dtype, source_type: str | None) -> str | None:
    if not source_type:
        return None
    match = DECIMAL.match(source_type.strip())
    if match is None or int(match.group(1)) <= WIDEST_EXACT_PRECISION:
        return None
    if dtype in DOUBLE_DTYPES:
        return "DOUBLE"
    if dtype in BIGINT_DTYPES:
        return "BIGINT"
    return None


def sample(cursor: Cursor, source: str, limit: int,
           projection: str = "*") -> None:
    from kumo_core.custom_sql import sample_select
    cursor.execute(sample_select(source, limit, projection))
