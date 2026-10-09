# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Optional

from kumo_core.training.options import allow_array_targets
from kumoapi.typing import ProblemType
from kumopql.parser import PQLParser as newPQLParser
from kumopql.validator import PredictiveQueryValidator

from kumo.config.graph import GraphConfig
from kumo.config.predictive_query import PredictiveQueryV2Config
from kumo.graph import Graph
from kumo.validation import ValidationException
from monitoring.tracing import TraceOptions, trace


class QueryDialect(str, Enum):
    YAML = 'YAML'
    PQLANG = 'PQLANG'
    PQLANG_V2 = 'PQLANG_V2'


@trace(TraceOptions(trace_params=True))
def parse_query_string_v2(
    name: str,
    graph: Graph,
    query_string: str,
    graph_config: Optional[GraphConfig] = None,
) -> PredictiveQueryV2Config:
    # TODO(jinu): add a pquery_str parameter to PredictiveQueryV2Config
    # don't split into entity, target and whatif in the config for pqueryv2
    from kumo.pquery.utils import validator_kwargs
    try:
        parsed_query = newPQLParser().to_parsed_predictive_query(query_string)
    except ValueError as error:
        # A grammar/parse failure is user input (a malformed PQL), not an
        # internal fault. Raise the allowlisted ValidationException so the
        # parser's actionable diagnostic surfaces on validate/model_plan
        # instead of being masked as a generic "contact Kumo support" 500.
        # `from None` keeps the raw query out of the exception chain, matching
        # the parser-input handling at rest/driver/pquery_logic.py:200-207.
        raise ValidationException(
            title="Invalid predictive query",
            message=str(error),
        ) from None
    validator = PredictiveQueryValidator(
        **validator_kwargs(graph, query_string=query_string))
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    if not response.ok:
        # Should not happen unless the config is corrupted
        raise ValidationException(
            title="Invalid predictive query",
            message=response.message(),
        )

    assert validated_query is not None
    num_forecasts = validated_query.num_forecasts
    forecast_length = (num_forecasts if validated_query.problem_type
                       == ProblemType.FORECAST else None)
    return PredictiveQueryV2Config(
        name=name,
        graph=graph_config or graph.config(),
        validated_predictive_query=validated_query,
        list_distinct_target_is_array=allow_array_targets(),
        forecast_length=forecast_length,
        lag_timesteps=1 if forecast_length is not None else None,
    )
