# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumoapi.json_serde import from_json, to_json
from kumoapi.pquery.AST import Column, Condition, Join


def test_missing_rhs_target():
    with pytest.raises(ValueError, match='Class \'Join\' is missing'):
        Join()


def test_join_serialization():
    condition = Condition(target=Join(rhs_target=Column(fqn='table.age')),
                          op='=', value=20)
    cfg = condition.to_string()
    assert cfg == "table.age = 20"
    cfg = to_json(condition)
    assert condition == from_json(cfg, Condition)


def test_all_join_columns():
    condition = Condition(
        target=Join(rhs_target=Column(fqn='table.age'), lhs_key='table.key1',
                    rhs_key='table2.key2'), op='=', value=20)
    assert sorted(condition.all_join_columns) == ['table.key1', 'table2.key2']
