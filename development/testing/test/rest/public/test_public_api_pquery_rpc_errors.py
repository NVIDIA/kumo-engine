# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from unittest.mock import AsyncMock

import kumoapi.graph as graph_api
import kumoapi.pquery as pquery_api
import kumoapi.source_table as source_table_api
import kumoapi.table as table_api
import pytest
from fastapi import HTTPException
from kumoapi.typing import Dtype, Stype

from kumo.validation import ValidationException
from rest.public.routers.pquery import PQueryRouter
from statedb.config.common import GraphIDv2, PredictiveQueryID
from statedb.utils.errors import (
    GraphDoesNotExistError,
    PredictiveQueryDoesNotExistError,
)
from temporalio_commons import RpcError

_PROJECT_ID = 'project_' + '1' * 32


def _minimal_public_pquery_resource() -> pquery_api.PQueryResource:
    return pquery_api.PQueryResource(
        query_string=(
            'PREDICT responses.is_sensitive FOR EACH responses.response_id'),
        graph=graph_api.GraphDefinition(
            tables={
                'responses':
                table_api.TableDefinition(
                    cols=[
                        table_api.Column(
                            name='response_id',
                            stype=Stype.ID,
                            dtype=Dtype.string,
                        ),
                        table_api.Column(
                            name='is_sensitive',
                            stype=Stype.categorical,
                            dtype=Dtype.bool,
                        ),
                    ],
                    pkey='response_id',
                    source_table=source_table_api.UnavailableSourceTable(
                        table='responses'),
                ),
            },
            col_groups=[],
        ),
        name='local_validation_exception_test',
    )


def _validation_rpc_error(method: str) -> RpcError:
    return RpcError(
        rpc_service='PQueryRpcService',
        rpc_method=method,
        error_type=ValidationException.__name__,
        message='Invalid predictive query',
        stack_trace='traceback omitted',
        non_retryable=True,
    )


def _not_found_rpc_error(method: str) -> RpcError:
    return RpcError(
        rpc_service='PQueryRpcService',
        rpc_method=method,
        error_type=PredictiveQueryDoesNotExistError.__name__,
        message='Predictive query missing_pquery does not exist',
        stack_trace='traceback omitted',
        non_retryable=True,
    )


@pytest.mark.asyncio
async def test_validate_pquery_maps_rpc_validation_exception_to_422():
    router = PQueryRouter.__new__(PQueryRouter)
    router._driver = AsyncMock()
    router._driver.validate_predictive_query_v2.side_effect = (
        _validation_rpc_error('ValidatePredictiveQuery'))

    with pytest.raises(HTTPException) as exc_info:
        await router.validate_pquery(
            _minimal_public_pquery_resource(),
            project_id=_PROJECT_ID,
        )

    assert exc_info.value.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert exc_info.value.detail == 'Invalid predictive query'


@pytest.mark.asyncio
async def test_get_pquery_maps_missing_predictive_query_to_404():
    router = PQueryRouter.__new__(PQueryRouter)
    router._driver = AsyncMock()
    router._driver.get_predictive_query_v2.side_effect = _not_found_rpc_error(
        'GetPredictiveQuery')

    with pytest.raises(HTTPException) as exc_info:
        await router.get_pquery(PredictiveQueryID('missing_pquery'),
                                _PROJECT_ID)

    assert exc_info.value.status_code == HTTPStatus.NOT_FOUND
    assert exc_info.value.detail == 'Predictive query missing_pquery does not exist'


@pytest.mark.asyncio
async def test_get_pquery_maps_direct_missing_predictive_query_to_404():
    router = PQueryRouter.__new__(PQueryRouter)
    router._driver = AsyncMock()
    router._driver.get_predictive_query_v2.side_effect = (
        PredictiveQueryDoesNotExistError(PredictiveQueryID('missing_pquery')))

    with pytest.raises(HTTPException) as exc_info:
        await router.get_pquery(PredictiveQueryID('missing_pquery'),
                                _PROJECT_ID)

    assert exc_info.value.status_code == HTTPStatus.NOT_FOUND
    assert exc_info.value.detail == (
        "Predictive Query with id='missing_pquery' does not exist")


@pytest.mark.asyncio
async def test_create_pquery_maps_rpc_validation_exception_to_422():
    router = PQueryRouter.__new__(PQueryRouter)
    router._driver = AsyncMock()
    router._driver.create_predictive_query_v2.side_effect = (
        _validation_rpc_error('CreatePredictiveQuery'))

    with pytest.raises(HTTPException) as exc_info:
        await router.create_pquery(
            _minimal_public_pquery_resource(),
            project_id=_PROJECT_ID,
        )

    assert exc_info.value.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert exc_info.value.detail == 'Invalid predictive query'


# A well-formed public graph id (``graph-<32 hex fpid>``) that does not exist:
_MISSING_GRAPH_ID = GraphIDv2('graph-' + '0' * 32)


@pytest.mark.asyncio
async def test_infer_task_type_maps_missing_graph_to_404():
    # Regression for kumo-ai/kumo#29814: infer_task_type was the only handler
    # in this router without wrap_driver_errors, so the driver's
    # GraphDoesNotExistError (a ValueError subclass, not an HTTPException)
    # escaped uncaught to ServerErrorMiddleware and became a raw 500 --
    # violating the 404 the route advertises. Pre-fix this raises
    # GraphDoesNotExistError (no HTTPException), so pytest.raises(HTTPException)
    # fails; post-fix the wrapper maps it to 404.
    router = PQueryRouter.__new__(PQueryRouter)
    router._driver = AsyncMock()
    router._driver.infer_task_type_pqlv2.side_effect = GraphDoesNotExistError(
        _MISSING_GRAPH_ID)

    with pytest.raises(HTTPException) as exc_info:
        await router.infer_task_type(
            pquery_string='PREDICT COUNT(orders.*, 0, 7, days) '
            'FOR EACH users.user_id',
            graph_id=_MISSING_GRAPH_ID,
            project_id=_PROJECT_ID,
        )

    assert exc_info.value.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.asyncio
async def test_infer_task_type_maps_bad_query_value_error_to_422():
    # The same wrapper also covers the malformed-query path: a parser ValueError
    # raised inside infer_task_type_pqlv2 for a bad PQL string (against a real
    # graph) maps to a clean 422 instead of a raw 500.
    router = PQueryRouter.__new__(PQueryRouter)
    router._driver = AsyncMock()
    router._driver.infer_task_type_pqlv2.side_effect = ValueError(
        'invalid predictive query syntax')

    with pytest.raises(HTTPException) as exc_info:
        await router.infer_task_type(
            pquery_string='PREDICT nonsense',
            graph_id=_MISSING_GRAPH_ID,
            project_id=_PROJECT_ID,
        )

    assert exc_info.value.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
