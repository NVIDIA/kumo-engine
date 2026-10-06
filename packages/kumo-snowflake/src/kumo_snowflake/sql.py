# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Mapping

from kumo_core.sources import sql
from kumo_core.sources.errors import PermanentSourceError
from kumoapi.typing import Dtype

QUOTE = '"'
MAX_IDENTIFIER_LENGTH = 255
TZ_AWARE_TYPES = frozenset({"TIMESTAMP_TZ", "TIMESTAMP_LTZ"})
#: Snowflake refuses to cast a ``TIME`` to a timestamp at all, so a column
#: declared this way is unloaded as it stands.
UNCONVERTIBLE_TYPES = frozenset({"TIME"})
TIMESTAMP_EXPRESSION = "CAST(CONVERT_TIMEZONE('UTC', {column}) AS TIMESTAMP_NTZ)"
NAIVE_EXPRESSION = "CAST({column} AS TIMESTAMP_NTZ)"


def quote_identifier(name: str) -> str:
    """Snowflake will not compile a longer identifier, so a name over the
    limit is refused here rather than by a statement that already ran."""
    if len(name) > MAX_IDENTIFIER_LENGTH:
        raise ValueError(
            f"a snowflake identifier holds at most {MAX_IDENTIFIER_LENGTH} "
            f"characters, but {name[:32]!r} runs to {len(name)}")
    return sql.quote_identifier(name, quote=QUOTE)


def qualified_name(*parts: str) -> str:
    return sql.qualified_name(*parts, quote=quote_identifier)


def build_projection(
    columns: Mapping[str, Dtype],
    aliases: Mapping[str, str] | None = None,
    source_types: Mapping[str, str] | None = None,
) -> str:
    """Snowflake cannot unload ``TIMESTAMP_TZ`` or ``TIMESTAMP_LTZ`` to
    Parquet, and a straight cast records two instants as one wall clock, so
    they are converted to UTC; converting a ``TIMESTAMP_NTZ`` instead shifts
    it by the session offset, so the declared type decides.
    """
    declared = source_types or {}

    def cast(name: str, dtype: Dtype, quoted: str) -> str:
        source = declared.get(name, "").upper()
        if source in TZ_AWARE_TYPES and dtype in {Dtype.time, Dtype.date}:
            return timestamp_ntz(quoted, source)
        if dtype is not Dtype.time:
            return quoted
        if source in UNCONVERTIBLE_TYPES:
            return quoted
        return timestamp_ntz(quoted, source)

    return sql.render_projection(columns, aliases, quote=quote_identifier,
                                 cast=cast)


#: A connection this package opens is ``pyformat``, but one borrowed from a
#: Snowpark session -- the only kind available inside Snowpark Container
#: Services -- is ``qmark``, and a query written for the wrong one reaches
#: the warehouse unsubstituted. Only position-independent styles are here:
#: ``numeric`` and ``named`` cannot repeat one marker across a query, so
#: rendering ``:1`` twice would bind both positions to the first value.
_PLACEHOLDERS = {"qmark": "?", "format": "%s", "pyformat": "%s"}


def placeholder(cursor: Any) -> str:
    """Read off the connection rather than assumed, because a reader is
    handed one it did not open."""
    style = getattr(getattr(cursor, "connection", None), "_paramstyle", None)
    if style is None:
        return _PLACEHOLDERS["pyformat"]
    if str(style) not in _PLACEHOLDERS:
        raise PermanentSourceError(
            f"this connection binds parameters {str(style)!r}, which names "
            f"each one individually; open it with 'qmark' or 'pyformat', "
            f"which is what both Snowpark and snowflake.connector do by "
            f"default")
    return _PLACEHOLDERS[str(style)]


def describe_columns(cursor: Any, query: str) -> list[tuple[str, Dtype]]:
    from kumo_snowflake.types import dtype_for_described
    from snowflake.connector.constants import FIELD_ID_TO_NAME

    metadata = cursor.describe(query)
    if metadata is None:
        metadata = cursor.description
    return [(column.name,
             dtype_for_described(FIELD_ID_TO_NAME[column.type_code],
                                 column.scale)) for column in metadata]


def timestamp_ntz(quoted: str, source_type: str) -> str:
    expression = (TIMESTAMP_EXPRESSION
                  if source_type in TZ_AWARE_TYPES else NAIVE_EXPRESSION)
    return expression.format(column=quoted)
