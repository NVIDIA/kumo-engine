# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import Callable, Mapping

from kumoapi.typing import Dtype

Quote = Callable[[str], str]
Cast = Callable[[str, Dtype, str], str]

#: A backslash is an escape in its own right in some dialects, and a line
#: break ends a statement in every one of them.
UNQUOTABLE = ("\\", "\n", "\r")
SNOWFLAKE_UNQUOTED_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*\Z")


def resolve_snowflake_identifier(name: str) -> str:
    r"""Resolve declared Snowflake syntax to the exact catalog name.

    Connector configuration owns the *declared* form (including delimiters),
    while catalog queries and qualified SQL own the exact name.  Keeping this
    conversion in the shared source layer gives the platform connector and
    the separately shipped Snowflake reader one boundary rule without making
    either package depend on the other.

    Unquoted Snowflake identifiers are stored uppercase.  Double-quoted
    identifiers preserve case, and a doubled quote inside them denotes one
    literal quote.
    """
    quote = '"'
    if not name:
        raise ValueError("a Snowflake identifier cannot be empty")
    if not name.startswith(quote):
        if SNOWFLAKE_UNQUOTED_IDENTIFIER.fullmatch(name) is None:
            raise ValueError(f'invalid unquoted Snowflake identifier: {name!r}')
        return name.upper()
    if len(name) < 2 or not name.endswith(quote):
        raise ValueError(f'unterminated quoted Snowflake identifier: {name!r}')
    body = name[1:-1]
    if not body:
        raise ValueError("a Snowflake identifier cannot be empty")
    parts = []
    index = 0
    while index < len(body):
        character = body[index]
        if character != quote:
            parts.append(character)
            index += 1
            continue
        if index + 1 >= len(body) or body[index + 1] != quote:
            raise ValueError(f'invalid quoted Snowflake identifier: {name!r}')
        parts.append(quote)
        index += 2
    return ''.join(parts)


def quote_identifier(name: str, *, quote: str, escape: bool = True) -> str:
    """Under ``escape=False`` the dialect has no in-band escape, so a name
    that could end the quoting is refused: nothing else stands between a
    schema-derived name and the statement.
    """
    if not name:
        raise ValueError("an identifier cannot be empty")
    if not escape:
        for character in (quote, *UNQUOTABLE):
            if character in name:
                raise ValueError(
                    f"identifier {name!r} cannot contain {character!r}")
        return f"{quote}{name}{quote}"
    return f"{quote}{name.replace(quote, quote * 2)}{quote}"


def qualified_name(*parts: str, quote: Quote) -> str:
    return ".".join(quote(part) for part in parts)


def validated_aliases(
    columns: Mapping[str, Dtype],
    aliases: Mapping[str, str] | None,
) -> Mapping[str, str]:
    """The *output* names are what has to be checked, not the aliases:
    renaming ``a`` to ``b`` beside a projected ``b`` collides just as two
    aliases onto one name do, and a warehouse that writes both leaves Parquet
    whose columns cannot be told apart.
    """
    if not columns:
        raise ValueError("a projection of at least one column is required")
    aliases = dict(aliases or {})
    unknown = set(aliases) - set(columns)
    if unknown:
        raise ValueError(
            f"aliases name columns not requested: {sorted(unknown)}")
    outputs = [aliases.get(name, name) for name in columns]
    if any(not name for name in outputs):
        raise ValueError("a projected column cannot be named nothing")
    collisions = sorted({name for name in outputs if outputs.count(name) > 1})
    if collisions:
        raise ValueError(
            f"the projection would name two columns alike: {collisions}")
    return aliases


def render_projection(
    columns: Mapping[str, Dtype],
    aliases: Mapping[str, str] | None = None,
    *,
    quote: Quote,
    cast: Cast | None = None,
) -> str:
    aliases = validated_aliases(columns, aliases)
    terms = []
    for name, dtype in columns.items():
        quoted = quote(name)
        expression = cast(name, Dtype(dtype), quoted) if cast else quoted
        alias = aliases.get(name, name)
        if expression != quoted or alias != name:
            expression = f"{expression} AS {quote(alias)}"
        terms.append(expression)
    return ", ".join(terms)
