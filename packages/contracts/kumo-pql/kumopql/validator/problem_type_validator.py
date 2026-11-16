# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumoapi.common import ValidationError, ValidationResponse
from kumoapi.pquery import ParsedPredictiveQuery
from kumoapi.typing import ProblemType, Stype

VALID_PROBLEM_TYPES = ['CLASSIFY', 'RANK', 'FORECAST']
MAX_TIMEFRAMES = 10_000


class ProblemTypeValidator:
    """All validation related to `problem_type` field
    in `PredictiveQuery` object.
    """
    def validate(
        self,
        query: ParsedPredictiveQuery,
    ) -> ValidationResponse:
        """Validates the `problem_type` field in `PredictiveQuery` object.
        Currently the only supported problem types are `CLASSIFY` and `RANK`.
        And the target should have stype `multicategorical`.

        Args:
            query: `ParsedPredictiveQuery` with stypes inferred.
        Note:
            This validator requires the stypes to be resolved before being
            called as it uses the stypes to validate the problem type.

        """
        problem_type = query.problem_type
        validation_response = ValidationResponse()
        target_ast = query.target_ast

        if target_ast.stype == Stype.multicategorical and problem_type is None:
            # If target is multicategorical and problem_type is not specified,
            # then we respond with an error.
            return ValidationResponse(errors=[
                ValidationError(
                    "Problem type is required for multicategorical targets.",
                    f"{query.entity_ast.get_location().message_start}: "
                    f"Since your target({target_ast}) is multicategorical, "
                    f"you must specify a problem type (CLASSIFY or RANK). "
                    f"Please rewrite the query as: PREDICT {target_ast} "
                    f"CLASSIFY/RANK TOP k",
                )
            ])
        if problem_type is None:
            return validation_response
        if problem_type == ProblemType.FORECAST:
            return self.validate_forecast(query)

        if problem_type.value.upper() not in VALID_PROBLEM_TYPES:
            return ValidationResponse(errors=[
                ValidationError(
                    f"Problem type '{problem_type}' is not supported. ",
                    f"Only {VALID_PROBLEM_TYPES} are supported problem types "
                    f"In PREDICT.",
                )
            ])

        assert target_ast.stype is not None
        if target_ast.stype != Stype.multicategorical:
            error_msg = (
                f"{query.target_ast.get_location().message_start}: "
                f"Target '{target_ast}' is not multicategorical. "
                f"Either drop the problem type {problem_type} OR change the "
                f"target to a multicategorical column, 'LAST'/'FIRST' "
                f"aggregation of a multicategorical column, "
                f"or a 'LIST_DISTINCT' aggregation.")
            return ValidationResponse(errors=[
                ValidationError(
                    f"Problem type '{problem_type}' is only supported for "
                    f"multicategorical targets. ",
                    error_msg,
                )
            ])

        # Rank problem requires top_k to be specified.
        if (problem_type == ProblemType.RANK and query.top_k is None):
            return ValidationResponse(errors=[
                ValidationError(
                    f"Problem type '{problem_type.value}' "
                    f"requires TOP K to be specified. ",
                    f"{query.entity_ast.get_location().message_start}: "
                    f"Please rewrite the query with 'TOP' as 'RANK TOP k'.",
                )
            ])
        if query.top_k is not None and query.top_k < 1:
            return ValidationResponse(errors=[
                ValidationError(
                    'Invalid TOP K value',
                    f'TOP K must be at least 1, got {query.top_k}.',
                )
            ])
        return validation_response

    def validate_forecast(
            self, parsed_query: ParsedPredictiveQuery) -> ValidationResponse:
        r"""Validates the FORECAST clause constraints.

        Rules:
            1. N must be >= 1 (N=1 is a single-horizon forecast).
            2. N must be <= MAX_TIMEFRAMES.
            3. Target must have a forward-looking time range.
            4. Target must be numerical.

        Args:
            parsed_query: the input parsed query.

        Returns:
            ValidationResponse: Errors and warnings encountered.
        """
        response = ValidationResponse()
        num_forecasts = parsed_query.num_forecasts
        if num_forecasts < 1:
            response.errors.append(
                ValidationError(
                    title='Invalid FORECAST value',
                    message=f'FORECAST value must be at least 1, '
                    f'got {num_forecasts}.'))
            return response

        if num_forecasts > MAX_TIMEFRAMES:
            response.errors.append(
                ValidationError(
                    title='FORECAST value too large',
                    message=f'FORECAST {num_forecasts} TIMEFRAMES exceeds '
                    f'the maximum of {MAX_TIMEFRAMES}.'))
            return response

        target_ast = parsed_query.target_ast
        has_time_range = target_ast.date_offset_range is not None
        if not has_time_range:
            response.errors.append(
                ValidationError(
                    title='FORECAST requires a temporal target',
                    message='FORECAST can only be used with targets that '
                    'have a forward-looking time range (e.g., '
                    'SUM(table.col, 0, 7, days)). Static column '
                    'targets are not supported with FORECAST.'))
            return response

        if target_ast.stype != Stype.numerical:
            response.errors.append(
                ValidationError(
                    title='FORECAST requires a numerical target',
                    message='FORECAST is only supported for numerical '
                    'targets.'))
            return response

        return response
