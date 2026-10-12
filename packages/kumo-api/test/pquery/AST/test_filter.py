# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumoapi.json_serde import from_json, to_json
from kumoapi.pquery.AST import Column, Condition, Filter


def test_missing_target():
    with pytest.raises(ValueError,
                       match='Class \'Filter\' is missing a target'):
        condition = Condition(target=Column(fqn='table.age'), op='>', value=20)
        Filter(condition=condition)


def test_missing_condition():
    with pytest.raises(ValueError,
                       match='Class \'Filter\' is missing a condition'):
        Filter(target=Column(fqn='table.value'))


def test_filter_serialization():
    condition = Condition(target=Column(fqn='table.age'), op='>', value=20)
    where = Filter(target=Column(fqn='table.value'), condition=condition)
    assert where.to_string() == 'table.value WHERE table.age > 20'
    assert where.to_string(
        rich=True) == ('table.value [bold]WHERE[/bold] table.age > 20')
    cfg = to_json(where)
    assert where == from_json(cfg, Filter)
