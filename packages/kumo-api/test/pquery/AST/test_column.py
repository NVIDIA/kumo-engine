# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

import pytest

from kumoapi.json_serde import from_json, to_json
from kumoapi.pquery.AST import Column
from kumoapi.pquery.AST.location_interval import ASTQueryLocationInterval


def test_invalid_column():
    with pytest.raises(ValueError, match='not given in its fully-qualified'):
        Column()
    with pytest.raises(ValueError, match='not given in its fully-qualified'):
        Column(fqn='a.b.c')


def test_column_serialization():
    column = Column(fqn='table.column')
    cfg = column.to_string()
    assert cfg == "table.column"
    cfg = to_json(column)
    assert column == from_json(cfg, Column)


def test_all_query_columns():
    column = Column(fqn='table.column')
    assert column.all_query_columns == ['table.column']


def test_all_query_columns_with_locations():
    column = Column(fqn='table.column',
                    location=ASTQueryLocationInterval(1, 2, 3, 4))
    assert column.all_query_columns_with_locations == [
        ('table.column', ASTQueryLocationInterval(1, 2, 3, 4))
    ]
