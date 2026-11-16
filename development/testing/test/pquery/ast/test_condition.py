# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumoapi.pquery.AST import ASTNode as apiASTNode
from kumopql.parser import PQLParser


@pytest.mark.parametrize('test_dict', [
    {
        'query': 'PREDICT (SUM(LOAN.AMOUNT, -30, 0) > 0) AND '
        '(AVG(ORDERS.AMOUNT, -60, 0) < 60) FOR EACH USER.ID',
        'target_all_columns': {'LOAN.AMOUNT', 'ORDERS.AMOUNT'},
        'entity_all_columns': {'USER.ID'},
        'target_all_location_msgs': {
            '(row 1, column 13: row 1, column 13)',
            '(row 1, column 48: row 1, column 48)'
        },
        'entity_all_location_msgs': {'(row 1, column 86: row 1, column 86)'},
    },
    {
        'query': 'PREDICT SUM(LOAN.AMOUNT, 0, 30, hours) '
        'FOR EACH USER.ID WHERE COUNT(LOAN.*, -30, 0) > 0',
        'target_all_columns': {'LOAN.AMOUNT'},
        'entity_all_columns': {'USER.ID', 'LOAN.*'},
        'target_all_location_msgs': {'(row 1, column 12: row 1, column 12)'},
        'entity_all_location_msgs': {
            '(row 1, column 68: row 1, column 68)',
            '(row 1, column 48: row 1, column 48)'
        },
    },
    {
        'query':
        'PREDICT SUM(LOAN.AMOUNT, 0, 3, hours) > 0 '
        'OR COUNT(LOAN.*, 1, 2, days) > 0 '
        'FOR EACH USER.ID WHERE COUNT(LOAN.* WHERE '
        'LAST(LOAN.STATUS, -5, -1) = "paid", -1, 0) > 0',
        'target_all_columns': {'LOAN.AMOUNT', 'LOAN.*'},
        'entity_all_columns': {'USER.ID', 'LOAN.*', 'LOAN.STATUS'},
        'target_all_location_msgs': {
            '(row 1, column 12: row 1, column 12)',
            '(row 1, column 51: row 1, column 51)'
        },
        'entity_all_location_msgs': {
            '(row 1, column 104: row 1, column 104)',
            '(row 1, column 122: row 1, column 122)',
            '(row 1, column 84: row 1, column 84)'
        },
    },
])
def test_all_query_columns(test_dict):
    parser = PQLParser()
    query = test_dict['query']
    parsed = parser.to_parsed_predictive_query(query)

    target = parsed.target_ast
    entity = parsed.entity_ast

    assert isinstance(target, apiASTNode)
    assert isinstance(entity, apiASTNode)

    target_all_columns = test_dict['target_all_columns']
    entity_all_columns = test_dict['entity_all_columns']

    assert set(target.all_query_columns) == target_all_columns
    assert set(entity.all_query_columns) == entity_all_columns

    all_query_cols = {x[0] for x in target.all_query_columns_with_locations}
    all_query_locations = {
        x[1].message
        for x in target.all_query_columns_with_locations
    }
    assert all_query_cols == target_all_columns
    assert all_query_locations == test_dict['target_all_location_msgs']

    all_query_cols = {x[0] for x in entity.all_query_columns_with_locations}
    all_query_locations = {
        x[1].message
        for x in entity.all_query_columns_with_locations
    }
    assert all_query_cols == entity_all_columns
    assert all_query_locations == test_dict['entity_all_location_msgs']
