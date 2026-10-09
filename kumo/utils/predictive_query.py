# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumopql.parser.parser import PQLParser
from kumopql.validator import PredictiveQueryValidator

from kumo.graph import Graph
from kumo.pquery.predictive_query import PredictiveQueryV2
from kumo.pquery.utils import validator_kwargs


def get_pquery(
    graph: Graph,
    query: str,
    name: str,
    **kwargs,
) -> PredictiveQueryV2:
    r"""Helper function to create a predictive query.

    Args:
        graph (Graph): A kumo Graph object.
        query (str): A predictive query syntax.
        name (str): The name of the returnd predictive query.

    Returns:
        (PredictiveQueryV2): The returned predictive query.
    """
    parser = PQLParser()
    validator = PredictiveQueryValidator(
        **validator_kwargs(graph, query_string=query))
    parsed_pquery = parser.to_parsed_predictive_query(query)
    valid_pquery, response = validator.validate_predictive_query(parsed_pquery)
    if not response.ok:
        raise ValueError(response.error_message)
    assert valid_pquery is not None
    return PredictiveQueryV2(
        name=name,
        graph=graph,
        validated_predictive_query=valid_pquery,
        **kwargs,
    )
