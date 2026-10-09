# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import kumoapi.model_plan as plan_api
import kumoapi.pquery as api
from fastapi import APIRouter, Depends, HTTPException
from fastapi.params import Query
from kumo_core.model_plan_conversion import public_model_plan_from_internal
from kumoapi.graph import GraphDefinition as ApiGraphDefinition
from kumoapi.task import TaskType
from typing_extensions import Annotated

from io_utils.json import from_json, from_json_dict, to_json, to_json_dict
from kumo.pquery.parser import QueryDialect
from kumo.validation import ValidationResponse
from rest.config.predictive_query_v2 import (
    PQueryResource,
    PQueryValidationRequest,
)
from rest.config.status import EmptyResponse, IDResponse
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.utils.public import wrap_driver_errors
from rest.utils.routers import wrap_validation_exceptions
from statedb.config.common import (
    _FP_SUFFIX_SEPARATOR,
    GraphIDv2,
    PredictiveQueryID,
    ProjectID,
)
from statedb.utils.errors import PredictiveQueryDoesNotExistError


def _internal_pquery_from_public(
        resource: api.PQueryResource) -> PQueryResource:
    r"""Build the internal DTO from the complete public field allowlist."""
    return PQueryResource(
        query_string=resource.query_string,
        graph=resource.graph,
        name=resource.name,
        desc=resource.desc,
    )


def _public_pquery_from_internal(
        resource: PQueryResource) -> api.PQueryResource:
    r"""Project an internal PQuery onto the exact public response fields."""
    return api.PQueryResource(
        query_string=resource.query_string,
        graph=from_json_dict(to_json_dict(resource.graph), ApiGraphDefinition),
        name=resource.name,
        desc=resource.desc,
    )


class PQueryRouter:
    def __init__(
        self,
        api_key_auth: Callable,
        driver: PQueryApiDriver,
        auth: Authentication,
    ) -> None:
        self._driver = driver
        self._auth = auth

        self.router = APIRouter(
            prefix="/predictive_queries",
            tags=["API v1 (Under Development): Predictive Queries"],
            dependencies=[Depends(api_key_auth)],
        )

        # CR[U]D ##############################################################

        # Create: either named (as template) or unnamed (as object keyed by
        #   FPID). Updating is always allowed directly via creation, of the
        #   same object multiple times. There is no validation done here, since
        #   updating amounts to updating a template.
        _create_responses: Dict[Union[int, str], Dict[str, Any]] = {
            HTTPStatus.BAD_REQUEST: {
                'description':
                "Cannot create a new predictive query as the query name "
                "already exists."
            },
            HTTPStatus.UNPROCESSABLE_ENTITY: {
                'description':
                "The predictive query definition is invalid, including when "
                "a required physical source is missing or unavailable."
            },
        }
        self.router.add_api_route(
            "",
            self.create_pquery,
            response_model=IDResponse,
            methods=['POST'],
            summary="Create a new predictive query with its full definition.",
            responses=_create_responses,
        )
        self.router.add_api_route(
            "/",
            self.create_pquery,
            response_model=IDResponse,
            methods=['POST'],
            summary="Create a new predictive query with its full definition.",
            responses=_create_responses,
            include_in_schema=False,
        )

        # Read: either from a name or an FPID
        self.router.add_api_route(
            "/{pquery_id}",
            self.get_pquery,
            response_model=api.PQueryResource,
            methods=['GET'],
            summary="Get a predictive query with its full definition.",
            responses={
                HTTPStatus.NOT_FOUND: {
                    'description':
                    "Unable to find the predictive query with the given name."
                }
            },
        )

        # Delete: by name only
        self.router.add_api_route(
            "/{pquery_id}",
            self.delete_pquery,
            response_model=EmptyResponse,
            methods=['DELETE'],
            summary="Delete a predictive query.",
            responses={
                HTTPStatus.NOT_FOUND: {
                    'description':
                    "Unable to find the predictive query with the given name."
                }
            },
        )

        # List: all named predictive queries (templates)
        self.router.add_api_route(
            "",
            self.list_pqueries,
            response_model=List[api.PQueryResource],
            methods=['GET'],
            summary="List predictive queries matching a name pattern.",
        )
        self.router.add_api_route(
            "/",
            self.list_pqueries,
            response_model=List[api.PQueryResource],
            methods=['GET'],
            summary="List predictive queries matching a name pattern.",
            include_in_schema=False,
        )

        # Utilities ###########################################################

        self.router.add_api_route(
            "/validate",
            self.validate_pquery,
            methods=['POST'],
            summary="Validates this predictive query.",
        )

        self.router.add_api_route(
            "/infer_task_type",
            self.infer_task_type,
            response_model=Tuple[TaskType, str],
            methods=['POST'],
            summary="Returns the task type of this predictive query.",
            responses={
                HTTPStatus.NOT_FOUND: {
                    'description': "Unable to find the graph with the given id."
                },
                HTTPStatus.UNPROCESSABLE_ENTITY: {
                    'description':
                    "The predictive query is semantically invalid for the "
                    "given graph."
                }
            },
        )

        self.router.add_api_route(
            '/model_plan',
            self.get_default_model_plan,
            response_model=plan_api.ModelPlanInfo,
            methods=['POST'],
            summary="Suggest default model plan for this predictive query.",
            responses={
                HTTPStatus.NOT_FOUND: {
                    'description':
                    "Unable to find the predictive query with the given name."
                }
            },
        )

        self.router.add_api_route(
            '/train_table_generation_plan',
            self.get_default_train_table_generation_plan,
            response_model=plan_api.TrainingTableGenerationPlan,
            methods=['POST'],
            summary=('Suggest default train-table generation plan for '
                     'this predictive query.'),
            responses={
                HTTPStatus.NOT_FOUND: {
                    'description':
                    "Unable to find the predictive query with the given name."
                }
            },
        )

    async def create_pquery(
        self,
        pquery_resource: api.PQueryResource,
        project_id: str,
        # Internal testing only.
        dialect: Annotated[
            QueryDialect,
            Query(include_in_schema=False)] = QueryDialect.PQLANG_V2,
    ) -> IDResponse:
        with wrap_driver_errors({KeyError: HTTPStatus.BAD_REQUEST}):
            pq_res = _internal_pquery_from_public(pquery_resource)
            pq_id = await self._driver.create_predictive_query_v2(
                pq_res,
                project_id=ProjectID(project_id),
            )
            return IDResponse(id=pq_id)

    async def get_pquery(
        self,
        pquery_id: PredictiveQueryID,
        project_id: str,
    ) -> api.PQueryResource:
        if _FP_SUFFIX_SEPARATOR in pquery_id:
            try:
                pquery_id = PredictiveQueryID.from_public(pquery_id)
            except BaseException as e:
                raise ValueError(f'Predictive query resource id is malformed '
                                 f': {pquery_id}') from e

        with (
                wrap_driver_errors(
                    {PredictiveQueryDoesNotExistError: HTTPStatus.NOT_FOUND}),
                wrap_validation_exceptions(PredictiveQueryDoesNotExistError),
        ):
            resource = await self._driver.get_predictive_query_v2(
                pquery_id, ProjectID(project_id))
            return _public_pquery_from_internal(resource)

    async def list_pqueries(
        self,
        project_id: str,
        name_pattern: Optional[str] = None,
    ) -> List[api.PQueryResource]:
        with wrap_driver_errors():
            driver = self._driver
            name_pattern = name_pattern or '.*'
            resources = await driver.list_predictive_queries_v2(
                ProjectID(project_id), name_pattern)
            return [
                _public_pquery_from_internal(resource) for resource in resources
            ]

    async def delete_pquery(
            self,
            pquery_id: PredictiveQueryID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> EmptyResponse:
        if PredictiveQueryID.has_fpid(pquery_id):
            raise HTTPException(
                HTTPStatus.FORBIDDEN,
                'Not allowed to delete a frozen unnamed pquery definition.')
        with wrap_driver_errors(
            {PredictiveQueryDoesNotExistError: HTTPStatus.NOT_FOUND}):
            await self._driver.delete_predictive_query(ctx, pquery_id,
                                                       ProjectID(project_id))
        return EmptyResponse()

    async def get_default_model_plan(
        self,
        request: plan_api.SuggestModelPlanRequest,
        project_id: str,
        # Internal testing only.
        dialect: Annotated[
            QueryDialect,
            Query(include_in_schema=False)] = QueryDialect.PQLANG_V2,
    ) -> plan_api.ModelPlanInfo:
        with wrap_driver_errors(
            {PredictiveQueryDoesNotExistError: HTTPStatus.NOT_FOUND}):
            has_weight_col = False
            if hasattr(request,
                       'train_table_spec') and request.train_table_spec:
                has_weight_col = request.train_table_spec.weight_col is not None  # noqa: E501
            plan_info = await self._driver.get_pquery_default_model_plan(
                request.query_string,
                GraphIDv2.from_public(request.graph_id),
                request.run_mode,
                ProjectID(project_id),
                has_weight_col,
            )

        return plan_api.ModelPlanInfo(
            model_plan=public_model_plan_from_internal(plan_info.model_plan),
            task_type=plan_info.task_type,
            query_type=plan_info.query_type,
            has_train_table_weight_col=plan_info.has_train_table_weight_col,
        )

    async def get_default_train_table_generation_plan(
        self,
        request: plan_api.SuggestModelPlanRequest,
        project_id: str,
        # Internal testing only.
        dialect: Annotated[
            QueryDialect,
            Query(include_in_schema=False)] = QueryDialect.PQLANG_V2,
    ) -> plan_api.TrainingTableGenerationPlan:
        with wrap_driver_errors(
            {PredictiveQueryDoesNotExistError: HTTPStatus.NOT_FOUND}):
            plan_info = await self._driver.get_pquery_default_model_plan(
                request.query_string, GraphIDv2.from_public(request.graph_id),
                request.run_mode, ProjectID(project_id))

        # Same as `get_default_model_plan`:
        return from_json(
            to_json(plan_info.model_plan.training_table_generation),
            plan_api.TrainingTableGenerationPlan)

    # Utilities ###############################################################

    async def infer_task_type(
        self,
        pquery_string: str,
        graph_id: GraphIDv2,
        project_id: Annotated[str, Query()],
    ) -> Tuple[TaskType, str]:
        """Return the inferred task type and reason.

        A query that is semantically invalid for the graph returns a 422
        validation error instead of a task-type tuple.
        """
        # TODO(manan): accept api.PQueryResource instead of the above explicit
        # parameters:
        with wrap_driver_errors():
            id = GraphIDv2.from_public(graph_id)
            out = await self._driver.infer_task_type_pqlv2(
                query=pquery_string,
                graph_id=id,
                project_id=ProjectID(project_id),
            )
            return (out.task_type, out.reason)

    async def validate_pquery(
        self,
        pquery_resource: api.PQueryResource,
        project_id: str,
        # Internal testing only.
        dialect: Annotated[
            QueryDialect,
            Query(include_in_schema=False)] = QueryDialect.PQLANG_V2,
    ) -> ValidationResponse:
        """Return semantic validation results for a predictive query.

        Missing or unavailable physical sources are reported in the response's
        validation errors rather than as opaque activity failures.
        """
        with wrap_driver_errors():
            pq_res = _internal_pquery_from_public(pquery_resource)
            return await self._driver.validate_predictive_query_v2(
                args=PQueryValidationRequest(pquery_resource=pq_res),
                project_id=ProjectID(project_id),
                check_unique_id=False,
            )
