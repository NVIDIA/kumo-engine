# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pandas as pd
import pytest
from kumoapi.pquery.AST import ASTNode as apiASTNode
from kumoapi.pquery.AST import DateOffsetRange as apiDateOffsetRange
from kumoapi.typing import TimeUnit
from kumopql.parser import PQLParser


@pytest.mark.parametrize(
    'test_dict',
    [
        {
            'query': 'PREDICT SUM(LOAN.AMOUNT, 0, 30, days) FOR EACH USER.ID',
            'target_range': apiDateOffsetRange(0, 30, TimeUnit.DAYS),
            'target_children_range': [None],
            'entity_range': None,
            'entity_children_range': [],
        },
        {
            'query': 'PREDICT SUM(LOAN.AMOUNT, 0, 30, hours) '
            'FOR EACH USER.ID WHERE COUNT(LOAN.*, -30, 0) > 0',
            'target_range': apiDateOffsetRange(0, 30, TimeUnit.HOURS),
            'target_children_range': [None],
            'entity_range': apiDateOffsetRange(-30, 0, TimeUnit.DAYS),
            'entity_children_range': [None],
        },
        {
            'query': 'PREDICT SUM(LOAN.AMOUNT, 0, 1800, minutes) '
            'FOR EACH USER.ID WHERE COUNT(LOAN.*, -30, 0) > 0',
            'target_range': apiDateOffsetRange(0, 1800, TimeUnit.MINUTES),
            'target_children_range': [None],
            'entity_range': apiDateOffsetRange(-30, 0, TimeUnit.DAYS),
            'entity_children_range': [None],
        },
        {
            'query':
            'PREDICT SUM(LOAN.AMOUNT, 0, 30) > 0 OR COUNT(LOAN.*, 15, 45) > 0 '
            'FOR EACH USER.ID WHERE COUNT(LOAN.*, -INF, 0) > 0',
            'target_range':
            apiDateOffsetRange(0, 45, TimeUnit.DAYS),
            'target_children_range': [
                apiDateOffsetRange(0, 30, TimeUnit.DAYS),
                apiDateOffsetRange(15, 45, TimeUnit.DAYS)
            ],  # noqa
            'entity_range':
            apiDateOffsetRange(None, 0, TimeUnit.DAYS),
            'entity_children_range':
            [None, apiDateOffsetRange(None, 0, TimeUnit.DAYS)],
        },
        {
            'query':
            'PREDICT SUM(LOAN.AMOUNT, 0, 3, hours) > 0 '
            'OR COUNT(LOAN.*, 1, 2, days) > 0 '
            'FOR EACH USER.ID WHERE COUNT(LOAN.* WHERE '
            'LAST(LOAN_STATUS.value, -5, -1) = "paid", -1, 0) > 0',
            'target_range':
            apiDateOffsetRange(0, 48, TimeUnit.HOURS),
            'target_children_range': [
                apiDateOffsetRange(0, 3, TimeUnit.HOURS),
                apiDateOffsetRange(24, 48, TimeUnit.HOURS)
            ],  # noqa
            'entity_range':
            apiDateOffsetRange(-5, 0, TimeUnit.DAYS),
            'entity_children_range':
            [None, apiDateOffsetRange(-5, 0, TimeUnit.DAYS)],
        },
    ])
def test_date_offset_range(test_dict):
    def normalize(offset: Optional[pd.DateOffset]) -> Optional[int]:
        if offset is None:
            return None
        dummy_date = pd.Timestamp('2000-01-01')
        normalized_date = dummy_date.normalize()  # type: ignore
        timedelta = normalized_date + offset - normalized_date
        return int(timedelta.total_seconds())

    query = test_dict['query']
    target_range = test_dict['target_range']
    target_children_range = test_dict['target_children_range']
    entity_range = test_dict['entity_range']
    entity_children_range = test_dict['entity_children_range']

    parser = PQLParser()
    parsed = parser.to_parsed_predictive_query(query)
    assert isinstance(parsed.entity_ast, apiASTNode)
    assert isinstance(parsed.target_ast, apiASTNode)
    assert parsed.entity_timeframe == entity_range
    assert parsed.target_timeframe == target_range

    entity = parsed.entity_ast
    target = parsed.target_ast

    assert entity.date_offset_range == entity_range
    assert target.date_offset_range == target_range

    if len(entity.children):
        for child, child_range in zip(entity.children, entity_children_range):
            if child_range is not None:
                assert (normalize(
                    child.date_offset_range.start_date_offset) == normalize(
                        child_range.start_date_offset))
                assert (normalize(
                    child.date_offset_range.end_date_offset) == normalize(
                        child_range.end_date_offset))

    if len(target.children):
        for child, child_range in zip(target.children, target_children_range):
            if child_range is not None:
                assert (normalize(
                    child.date_offset_range.start_date_offset) == normalize(
                        child_range.start_date_offset))
                assert (normalize(
                    child.date_offset_range.end_date_offset) == normalize(
                        child_range.end_date_offset))


@pytest.mark.parametrize('test_dict', [
    {
        'query': 'PREDICT SUM(LOAN.AMOUNT, 0, 30, days) FOR EACH USER.ID',
        'target_range': apiDateOffsetRange(0, 30, TimeUnit.DAYS),
        'entity_range': None,
    },
    {
        'query': 'PREDICT SUM(LOAN.AMOUNT, 0, 30, hours) '
        'FOR EACH USER.ID WHERE COUNT(LOAN.*, -30, 0) > 0',
        'target_range': apiDateOffsetRange(0, 30, TimeUnit.HOURS),
        'entity_range': apiDateOffsetRange(-30, 0, TimeUnit.DAYS),
    },
    {
        'query': 'PREDICT SUM(LOAN.AMOUNT, 0, 1800, minutes) '
        'FOR EACH USER.ID WHERE COUNT(LOAN.*, -30, 0) > 0',
        'target_range': apiDateOffsetRange(0, 1800, TimeUnit.MINUTES),
        'entity_range': apiDateOffsetRange(-30, 0, TimeUnit.DAYS),
    },
    {
        'query':
        'PREDICT SUM(LOAN.AMOUNT, 0, 30) > 0 OR COUNT(LOAN.*, 15, 45) > 0 '
        'FOR EACH USER.ID WHERE COUNT(LOAN.*, -INF, 0) > 0 '
        'OR COUNT(LOAN.*, -10, -5) < 10',
        'target_range':
        apiDateOffsetRange(0, 45, TimeUnit.DAYS),
        'entity_range':
        apiDateOffsetRange(-10, -5, TimeUnit.DAYS),
    },
    {
        'query': 'PREDICT SUM(LOAN.AMOUNT, 0, 30) '
        'FOR EACH USER.ID WHERE COUNT(LOAN.*, -INF, 0) > 0 ',
        'target_range': apiDateOffsetRange(0, 30, TimeUnit.DAYS),
        'entity_range': None,
    },
    {
        'query':
        'PREDICT SUM(LOAN.AMOUNT, 0, 3, hours) > 0 '
        'OR COUNT(LOAN.*, 1, 2, days) > 0 '
        'FOR EACH USER.ID WHERE COUNT(LOAN.* WHERE '
        'LAST(LOAN_STATUS.value, -5, -1) = "paid", -INF, 0) > 0',
        'target_range':
        apiDateOffsetRange(0, 48, TimeUnit.HOURS),
        'entity_range':
        apiDateOffsetRange(-5, -1, TimeUnit.DAYS),
    },
])
def test_non_inf_date_offset_range(test_dict):
    query = test_dict['query']
    target_range = test_dict['target_range']
    entity_range = test_dict['entity_range']

    parser = PQLParser()
    parsed = parser.to_parsed_predictive_query(query)
    assert isinstance(parsed.entity_ast, apiASTNode)
    assert isinstance(parsed.target_ast, apiASTNode)

    entity = parsed.entity_ast
    target = parsed.target_ast

    assert entity.non_inf_date_offset_range() == entity_range
    assert target.non_inf_date_offset_range() == target_range
