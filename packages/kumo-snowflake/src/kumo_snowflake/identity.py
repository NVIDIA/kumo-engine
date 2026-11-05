# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Iterable, Mapping

from kumo_core.sources.errors import PermanentSourceError, SourceNotFoundError
from kumo_core.sources.identity import (
    SourceVersion,
    epoch_second,
    hour_version,
)
from kumo_core.sources.sql import resolve_snowflake_identifier
from kumo_snowflake.sql import placeholder, quote_identifier

VIEW_TYPES = frozenset({"VIEW", "MATERIALIZED VIEW"})

ALTERED_QUERY = ("select TABLE_NAME, LAST_ALTERED from {database}"
                 ".INFORMATION_SCHEMA.TABLES "
                 "where TABLE_SCHEMA = {mark} and TABLE_NAME = {mark}")


def resolved_table_name(
    cursor: Any,
    *,
    database: str,
    schema: str,
    table: str,
) -> str:
    r"""Return the catalog table named by one provider-native coordinate.

    SDK discovery returns exact catalog names, including lowercase quoted
    names, so the exact spelling wins.  A caller may also supply Snowflake's
    declared, unquoted spelling; if the exact spelling is absent, its folded
    catalog name is tried in the same query.  This keeps ``list_tables()`` →
    ``table(name)`` lossless while making a direct ``orders`` read find the
    ordinary unquoted ``ORDERS`` table.
    """
    candidates = [table]
    try:
        folded = resolve_snowflake_identifier(table)
    except ValueError:
        # Exact provider-native names may contain characters that require
        # delimiters in SQL.  They are values in this catalog query, not SQL
        # syntax, so retain them; the shared resolver still rejects the same
        # value wherever declared identifier syntax is required.
        folded = table
    if folded != table:
        candidates.append(folded)

    mark = placeholder(cursor)
    marks = ", ".join(mark for _ in candidates)
    cursor.execute(
        ("select TABLE_NAME from "
         f"{quote_identifier(database)}.INFORMATION_SCHEMA.TABLES "
         f"where TABLE_SCHEMA = {mark} and TABLE_NAME in ({marks})"),
        (schema, *candidates),
    )
    found = {str(row[0]) for row in cursor.fetchall()}
    for candidate in candidates:
        if candidate in found:
            return candidate
    raise SourceNotFoundError(f"snowflake has no {schema}.{table} to read",
                              table=table)


def is_view(cursor: Any, *, database: str, schema: str, table: str) -> bool:
    """Materialized views are absent from ``INFORMATION_SCHEMA.VIEWS``, so
    the type is read from the table listing that carries both kinds."""
    from kumo_snowflake.catalog import table_type
    kind = table_type(cursor, database=database, schema=schema, table=table)
    if kind is None:
        raise SourceNotFoundError(f"snowflake has no {schema}.{table} to stamp",
                                  table=table)
    return kind in VIEW_TYPES


def data_version(
    cursor: Any,
    *,
    database: str,
    schema: str,
    table: str,
) -> str:
    """``LAST_ALTERED`` alone, without the ``ROW_COUNT`` the SDK's own stamp
    paired it with, so two edits within the catalog's resolution share a
    version; pin with Time Travel where that matters. Pandas parses the
    timestamp because a naive value it reads as UTC and the standard library
    reads as local time.
    """
    import pandas as pd
    if is_view(cursor, database=database, schema=schema, table=table):
        return hour_version()
    cursor.execute(
        ALTERED_QUERY.format(database=quote_identifier(database),
                             mark=placeholder(cursor)), (schema, table))
    altered = {str(row[0]): row[1] for row in cursor.fetchall()}
    if table not in altered:
        return hour_version()
    return SourceVersion(epoch_second(pd.to_datetime(altered[table])),
                         "timestamp")


class MissingColumnTypesError(PermanentSourceError):
    def __init__(self, columns: Iterable[str]) -> None:
        self.columns = tuple(sorted(columns))
        super().__init__(
            f"snowflake reported no type for {list(self.columns)}, whose "
            f"timezone semantics decide how they must be cast")


def column_types(cursor: Any, source: str,
                 columns: Iterable[str]) -> Mapping[str, str]:
    """``describe`` compiles without executing, so the types are free. A name
    is matched case-insensitively only where that is unambiguous, since two
    quoted identifiers may differ by case alone.
    """
    from snowflake.connector.constants import FIELD_ID_TO_NAME

    wanted = set(columns)
    if not wanted:
        return {}
    metadata = cursor.describe(f"SELECT * FROM {source}")
    if metadata is None:
        metadata = cursor.description
    declared = {
        column.name: FIELD_ID_TO_NAME[column.type_code]
        for column in metadata
    }
    folded: dict[str, list[str]] = {}
    for name, type_name in declared.items():
        folded.setdefault(name.upper(), []).append(type_name)

    resolved = {}
    for name in wanted:
        if name in declared:
            resolved[name] = declared[name]
        elif len(folded.get(name.upper(), [])) == 1:
            resolved[name] = folded[name.upper()][0]
    missing = wanted - set(resolved)
    if missing:
        raise MissingColumnTypesError(missing)
    return resolved
