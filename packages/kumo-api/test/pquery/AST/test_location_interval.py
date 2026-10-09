# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kumoapi.json_serde import from_json, to_json
from kumoapi.pquery.AST.location_interval import ASTQueryLocationInterval


def test_config():
    sample_interval = ASTQueryLocationInterval(1, 1, 3, 4)
    cfg = to_json(sample_interval)
    assert sample_interval == from_json(cfg, ASTQueryLocationInterval)
    sample_interval = ASTQueryLocationInterval(1, 1, 3, 4, False)
    cfg = to_json(sample_interval)
    assert sample_interval == from_json(cfg, ASTQueryLocationInterval)


@pytest.mark.parametrize(
    'args1,args2,exp_outcome',
    [
        ((2, 3, 2, 5), (2, 5, 2, 6), (2, 3, 2, 6)),  # All on same row
        ((2, 5, 2, 6), (2, 3, 2, 5), (2, 3, 2, 6)),  # All on same row, swapped
        ((2, 5, 2, 6), (3, 3, 3, 5), (2, 5, 3, 5)),  # Two separate rows
        ((3, 3, 3, 5), (2, 5, 2, 6),
         (2, 5, 3, 5)),  # Two separate rows, swapped
        ((2, 5, 2, 6), (3, 3, 3, 5, False), (2, 5, 2, 6)),  # Invalid data 1
        ((2, 5, 2, 6, False), (3, 3, 3, 5), (3, 3, 3, 5)),  # Invalid data 2
    ])
def test_merge(args1, args2, exp_outcome):
    interval1 = ASTQueryLocationInterval(*args1)
    interval2 = ASTQueryLocationInterval(*args2)
    exp_outcome_interval = ASTQueryLocationInterval(*exp_outcome)
    assert exp_outcome_interval == ASTQueryLocationInterval.merge(
        interval1, interval2)


def test_message():
    sample_interval = ASTQueryLocationInterval(1, 1, 3, 4)
    assert sample_interval.message == '(row 1, column 1: row 3, column 4)'


def test_message_unavailable():
    sample_interval = ASTQueryLocationInterval(1, 1, 3, 4, False)
    assert sample_interval.message == ''
