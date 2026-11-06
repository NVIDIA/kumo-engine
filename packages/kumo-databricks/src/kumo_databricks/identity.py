# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Mapping

from databricks.sql.client import Cursor
from kumo_core.sources.errors import SourceNotFoundError
from kumo_core.sources.identity import (
    SourceVersion,
    epoch_second,
    hour_version,
)
from kumo_databricks.errors import is_not_found

DESCRIBE = "DESCRIBE EXTENDED {source}"
DESCRIBE_QUERY = "DESCRIBE QUERY SELECT * FROM {source}"
SECTION_MARKER = "#"
HISTORY = "DESCRIBE HISTORY {source} LIMIT 1"


def column_types(cursor: Cursor, source: str) -> Mapping[str, str]:
    """``DESCRIBE EXTENDED`` answers with the column list, then sections
    under heading rows; a partitioned table repeats its partition columns in
    one of them, so collection stops at the first heading."""
    cursor.execute(DESCRIBE.format(source=source))
    types: dict[str, str] = {}
    for row in cursor.fetchall():
        name = str(row[0])
        dtype = str(row[1]).strip()
        if not dtype and (not name.strip() or name.startswith(SECTION_MARKER)):
            break
        types[name] = dtype
    return types


def query_column_types(cursor: Cursor, source: str) -> Mapping[str, str]:
    """Declared column types of a query. ``DESCRIBE EXTENDED`` does not
    accept one; ``DESCRIBE QUERY`` answers with the column list alone."""
    return dict(query_columns(cursor, source))


def query_columns(cursor: Cursor, source: str) -> list[tuple[str, str]]:
    cursor.execute(DESCRIBE_QUERY.format(source=source))
    return [(str(row[0]), str(row[1])) for row in cursor.fetchall()
            if str(row[0]).strip()]


def data_version(cursor: Cursor, source: str) -> str:
    """The commit time rides alongside the Delta version because a table
    dropped and recreated under one name restarts its history at zero. A view
    has no log, so it takes the hour window.
    """
    try:
        return table_version(cursor, source)
    except Exception as error:
        if is_not_found(error):
            raise SourceNotFoundError(f"databricks has no {source} to stamp",
                                      table=source) from None
    return hour_version()


def table_version(cursor: Cursor, source: str) -> str:
    import pandas as pd

    cursor.execute(HISTORY.format(source=source))
    rows = cursor.fetchall()
    if rows:
        commit = epoch_second(pd.Timestamp(rows[0][1]))
        return SourceVersion(f"{rows[0][0]}_{commit}", "exact")
    return hour_version()
