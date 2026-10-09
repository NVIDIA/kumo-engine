# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumo.pquery.weight_spec import WeightSpec, parse_weight_spec
from kumo.validation import ValidationException


def test_parse_weight_spec_column():
    spec = parse_weight_spec("transactions.weight")
    assert spec == WeightSpec(column="transactions.weight", aggregation=None,
                              raw="transactions.weight", is_expression=False)


def test_parse_weight_spec_none():
    assert parse_weight_spec(None) is None


@pytest.mark.parametrize(
    "expr,error_pattern",
    [
        ("", r"Weight column cannot be empty\."),
        ("ANY()", r"Weight column must be in table\.column form\."),
        ("ANY(*)", r"Weight column must be in table\.column form\."),
        ("ANY( )", r"Weight column must be in table\.column form\."),
        ("ANY(weight", r"Weight column must be in table\.column form\."),
        (".weight", r"Weight column must be in table\.column form\."),
        ("transactions.", r"Weight column must be in table\.column form\."),
    ],
)
def test_parse_weight_spec_invalid_expr(expr, error_pattern):
    with pytest.raises(ValidationException, match=error_pattern):
        parse_weight_spec(expr)


@pytest.mark.parametrize(
    "expr",
    [
        "weight",
        "ANY(weight)",
        "table.subtable.col",
    ],
)
def test_parse_weight_spec_requires_table_qualified_column(expr):
    with pytest.raises(ValidationException,
                       match=r"Weight column must be in table\.column form\."):
        parse_weight_spec(expr)
