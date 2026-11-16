# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from kumopql.parser import PQLParser
from kumopql.validator import PredictiveQueryValidator

from kumo.pquery.predictive_query import PredictiveQueryV2
from kumo.pquery.train_table_generator import (
    _check_diskgraph_downsampling_parity,
)
from kumo.pquery.utils import validator_kwargs
from statedb.utils.errors import UnsupportedInDiskgraphError


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
@patch('kumo.pquery.utils.get_max_train_rows', new=lambda: 6)
@patch(
    'kumo.graph.column.EncoderGenerator.'
    'MAX_MULTI_LABEL_CATEGORIES', new=1)
def test_generate_train_table_array_downsampling(
    create_lp_graph,
    tempdir,
):
    graph = create_lp_graph(tempdir.name)
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID, 0, 7) RANK TOP 10 '
             'FOR EACH USERS.USER_ID')
    parser = PQLParser()
    validator = PredictiveQueryValidator(**validator_kwargs(graph))
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert validated_query is not None
    assert response.ok

    predictive_query = PredictiveQueryV2(
        name='downsample_test', graph=graph,
        validated_predictive_query=validated_query)
    assert predictive_query.config() is not None

    with pytest.raises(UnsupportedInDiskgraphError,
                       match='DISKGRAPH_PRODUCER_DOWNSAMPLING_ENABLED'):
        _check_diskgraph_downsampling_parity(True)
