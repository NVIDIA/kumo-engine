# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumopql.parser import PQLParser


@pytest.mark.parametrize("query", [
    'PREDICT LOAN.STATUS FOR EACH LOAN.LOAN_ID',
    'PREDICT LOAN.STATUS FOR EACH LOAN.LOAN_ID ASSUMING LOAN.STATUS = "paid"',
    'PREDICT LOAN.STATUS FOR EACH LOAN.LOAN_ID WHERE LOAN.STATUS = "paid"',
    ('PREDICT SUM(LOAN.AMOUNT, 0, 30) FOR EACH USER.ID',
     'PREDICT SUM(LOAN.AMOUNT, 0, 30, days) FOR EACH USER.ID'),
    'PREDICT SUM(LOAN.AMOUNT, 0, 30, days) FOR EACH USER.ID',
    ('PREDICT SUM(LOAN.AMOUNT WHERE LOAN.STATUS = "paid", 0, 30, days) '
     'FOR EACH USER.ID'),
    'PREDICT LIST_DISTINCT(LOAN.STATUS) FOR EACH USER.ID',
    'PREDICT LIST_DISTINCT(LOAN.STATUS, 0, 30, days) FOR EACH USER.ID',
    ('PREDICT LIST_DISTINCT(LOAN.STATUS, 0, 30, days) RANK TOP 1 '
     'FOR EACH USER.ID WHERE LOAN.STATUS = "paid"'),
    ('PREDICT LIST_DISTINCT(LOAN.STATUS, 0, 30, days) CLASSIFY TOP 1 '
     'FOR EACH USER.ID WHERE LOAN.STATUS = "paid"'),
])
def test_string_conversion(query):
    pq_parser = PQLParser()
    original_query = query[0] if isinstance(query, tuple) else query
    pq = pq_parser.to_parsed_predictive_query(original_query)
    expected_string = query[1] if isinstance(query, tuple) else original_query
    assert pq.to_string() == expected_string
