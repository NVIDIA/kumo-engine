# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every dialect renders identifiers and projections through one function."""

import pytest
from kumo_core.sources.sql import (
    qualified_name,
    quote_identifier,
    render_projection,
    resolve_snowflake_identifier,
    validated_aliases,
)
from kumoapi.typing import Dtype

COLUMNS = {"id": Dtype.int64, "seen_at": Dtype.time}
REVERSED = {"seen_at": Dtype.time, "id": Dtype.int64}


def backticked(name):
    return quote_identifier(name, quote="`")


def widen(name, dtype, quoted):
    return f"CAST({quoted} AS DOUBLE)" if dtype is Dtype.time else quoted


class TestIdentifiers:
    def test_a_dialects_own_quote_is_doubled_inside_a_name(self):
        assert quote_identifier('od"d', quote='"') == '"od""d"'
        assert backticked("od`d") == "`od``d`"

    @pytest.mark.parametrize("name",
                             ["a`, (SELECT 1) AS `b", "a\\b", "a\nb", "a\rb"])
    def test_a_dialect_with_no_escape_refuses_the_name_instead(self, name):
        """A schema-derived name reaching an unescapable quote is injection,
        and the backslash and the line breaks end it as surely as the quote."""
        with pytest.raises(ValueError):
            quote_identifier(name, quote="`", escape=False)

    def test_an_empty_identifier_is_refused(self):
        with pytest.raises(ValueError):
            backticked("")

    def test_a_qualified_name_quotes_every_part(self):
        assert qualified_name("a", "b", quote=backticked) == "`a`.`b`"

    @pytest.mark.parametrize("declared,exact", [
        ("analytics", "ANALYTICS"),
        ('"analytics"', "analytics"),
        ('"an""alytics"', 'an"alytics'),
        ('"O\'Reilly"', "O'Reilly"),
    ])
    def test_snowflake_declarations_resolve_to_exact_catalog_names(
            self, declared, exact):
        assert resolve_snowflake_identifier(declared) == exact

    @pytest.mark.parametrize("declared", [
        "", '""', '"unterminated', '"a"b"', "bad-name", "bad name", 'bad"name',
        "1bad"
    ])
    def test_invalid_snowflake_identifier_syntax_is_refused(self, declared):
        with pytest.raises(ValueError):
            resolve_snowflake_identifier(declared)


@pytest.mark.parametrize("columns,aliases", [(COLUMNS, {
    "missing": "x"
}), ({}, None)])
def test_a_projection_that_is_empty_or_renames_a_stranger_is_refused(
        columns, aliases):
    with pytest.raises(ValueError):
        validated_aliases(columns, aliases)


@pytest.mark.parametrize("aliases", [{
    "id": "seen_at"
}, {
    "id": "key",
    "seen_at": "key"
}])
def test_a_projection_naming_two_columns_alike_is_refused(aliases):
    with pytest.raises(ValueError, match="two columns alike"):
        validated_aliases(COLUMNS, aliases)


def test_a_projection_naming_a_column_nothing_is_refused():
    with pytest.raises(ValueError):
        validated_aliases(COLUMNS, {"id": ""})


@pytest.mark.parametrize("columns,aliases,cast,rendered", [
    (COLUMNS, None, None, "`id`, `seen_at`"),
    (COLUMNS, {
        "id": "customer"
    }, None, "`id` AS `customer`, `seen_at`"),
    (COLUMNS, None, widen, "`id`, CAST(`seen_at` AS DOUBLE) AS `seen_at`"),
    (COLUMNS, {
        "seen_at": "ts"
    }, widen, "`id`, CAST(`seen_at` AS DOUBLE) AS `ts`"),
    (REVERSED, None, None, "`seen_at`, `id`"),
])
def test_the_select_list_names_every_column_the_graph_expects(
        columns, aliases, cast, rendered):
    assert render_projection(columns, aliases, quote=backticked,
                             cast=cast) == rendered
