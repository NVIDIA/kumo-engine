# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from unittest.mock import MagicMock

from kumoapi.pquery import ValidatedPredictiveQuery
from kumopql.parser.parser import PQLParser
from kumopql.validator import PredictiveQueryValidator

from kumo.graph import Graph
from kumo.pquery.train_table_generator import TrainingTableGeneratorV2
from kumo.pquery.utils import validator_kwargs
from kumo.validation.validation import ValidationResponse


def get_validated_query(
    graph: Graph,
    query: str,
) -> ValidatedPredictiveQuery:
    parser = PQLParser()
    validator = PredictiveQueryValidator(**validator_kwargs(graph))
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok, response
    assert validated_query is not None
    return validated_query


@contextmanager
def disable_pre_validation():
    original_func = TrainingTableGeneratorV2.validate_pre_train_table_generate
    TrainingTableGeneratorV2.validate_pre_train_table_generate = MagicMock(
        side_effect=lambda **kwargs: ValidationResponse())
    try:
        yield
    finally:
        TrainingTableGeneratorV2.validate_pre_train_table_generate = \
            original_func
