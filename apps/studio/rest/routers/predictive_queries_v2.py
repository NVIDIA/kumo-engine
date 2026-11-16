# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Dict, Optional, Tuple, Union

import kumoapi.model_plan as plan_api
from fastapi import APIRouter, Depends, HTTPException
from fastapi.params import Query
from kumo_core.types import TaskType
from kumoapi import jobs as jobs_api
from kumopql.rewriting.bp_filter_transformer import EditableQuery
from typing_extensions import Annotated

from kumo.config.model_plan_yaml import model_plan_to_yaml
from kumo.pquery.parser import QueryDialect
from rest.config.predictive_query import (
    InsertFilterOverridesRequest,
    OutputPresignedUrlResponse,
)
from rest.config.predictive_query_v2 import (
    ModelPlanResponse,
    PQueryEncodersResponse,
    PQueryMetadataResponse,
    PQueryResource,
    PQueryResponse,
    PQueryValidationRequest,
)
from rest.config.status import IDResponse
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.driver.projects_driver import ProjectsDriver
from rest.utils.public import wrap_driver_errors
from rest.utils.routers import nullable_response
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.utils.errors import (
    PredictiveQueryDoesNotExistError,
    TrainingJobDoesNotExistError,
)


class PredictiveQueryV2Router:
    """
    Router for handling predictive query operations in v2 API.
    This router provides endpoints for creating and managing predictive
    queries, including suggesting model plans based on provided queries.
    """
    def __init__(
        self,
        auth: Authentication,
        driver: PQueryApiDriver,
        projects_driver: ProjectsDriver,
    ) -> None:
        self._driver = driver
        self._auth = auth
        self._projects_driver = projects_driver

        self.router = APIRouter(
            prefix="/v1/predictive_queries",
            tags=["API v1 (Under Development): Predictive Queries"],
            dependencies=[Depends(auth.authenticate_request_header)],
        )

        _create_pquery_responses = {
            HTTPStatus.UNPROCESSABLE_ENTITY: {
                'description':
                "Unable to create a predictive query with the provided "
                "inputs"
            }
        }
        self.router.add_api_route(
            "/",
            self.create_pquery_v2,
            response_model=IDResponse,
            methods=['POST'],
            summary="Create a new predictive query with its full definition.",
            responses=_create_pquery_responses,
            include_in_schema=False,
        )

        self.router.add_api_route(
            "/{id}",
            self.get_pquery_v2,
            response_model=PQueryResponse,
            methods=['GET'],
            summary="Get a predictive query by ID.",
            responses={
                HTTPStatus.NOT_FOUND: {
                    'description': "Predictive query not found."
                }
            },
        )

        self.router.add_api_route(
            "/{id}/metadata",
            self.get_pquery_metadata_v2,
            response_model=PQueryMetadataResponse,
            methods=['GET'],
            summary="Get the task type and query type for a predictive "
            "query.",
            responses={
                HTTPStatus.NOT_FOUND: {
                    'description': "Predictive query not found."
                }
            },
        )

        self.router.add_api_route(
            "/{id}/download_holdout",
            self.generate_holdout_presigned_url_v2,
            response_model=OutputPresignedUrlResponse,
            methods=['GET'],
            summary="Generate a presigned URL for the holdout dataset.",
            responses={
                HTTPStatus.NOT_FOUND: {
                    'description': "Predictive query not found."
                },
                HTTPStatus.BAD_REQUEST: {
                    'description': "Predictive query has not been trained yet."
                },
            },
        )

        self.router.add_api_route(
            "/{id}/encoders",
            self.get_encoders_for_pquery_v2,
            response_model=Optional[PQueryEncodersResponse],
            methods=['GET'],
            summary="Get the encoders used for a predictive query.",
            responses=nullable_response(),
        )

        self.router.add_api_route(
            "/get_many",
            self.get_many_pqueries_v2,
            response_model=Dict[Union[TrainingJobID, BatchPredictionJobID,
                                      GeneratePredictionTableJobID,
                                      GenerateTrainTableJobID],
                                Tuple[str, TaskType]],
            methods=['POST'],
            summary="Get predictive query string for each job.",
        )

        self.router.add_api_route(
            "/model_plan",
            self.get_default_model_plan,
            response_model=ModelPlanResponse,
            methods=['POST'],
            summary="Generate a model plan for a predictive query.",
            responses={
                HTTPStatus.UNPROCESSABLE_ENTITY: {
                    'description':
                    "Unable to generate a model plan for the provided query."
                }
            },
        )

        self.router.add_api_route(
            "/validate",
            self.validate_pquery,
            response_model=jobs_api.ErrorDetails,
            methods=['POST'],
            summary="Validates this predictive query.",
        )

        self.router.add_api_route(
            "/{training_job_id}/get_allowed_bp_query_overrides",
            self.get_allowed_bp_query_overrides_v2,
            response_model=EditableQuery,
            methods=['GET'],
            summary="Get allowed batch prediction query overrides for a V2"
            " training job.",
        )

        self.router.add_api_route(
            "/{training_job_id}/insert_filter_overrides",
            self.insert_filter_overrides,
            response_model=EditableQuery,
            methods=['POST'],
            summary="Insert filter overrides for a training job.",
            responses={
                HTTPStatus.BAD_REQUEST: {
                    'description': "Invalid filter overrides provided."
                }
            },
        )

    async def create_pquery_v2(
        self,
        request: PQueryResource,
        project_id: Annotated[str, Query()],
    ) -> IDResponse:
        """
        Create a new predictive query.

        Args:
            request (plan_api.CreatePQueryRequest): The request object
            containing the query details.

        Returns:
            IDResponse: The response object containing the ID of the created
            predictive query.
        """
        await self._projects_driver._validate_project_for_creation(project_id)

        # We do not support older versions of PQL in v1 APIs.
        with wrap_driver_errors({KeyError: HTTPStatus.BAD_REQUEST}):
            pq_id = await self._driver.create_predictive_query_v2(
                request,
                project_id=ProjectID(project_id),
            )
            return IDResponse(id=pq_id)

    async def get_pquery_v2(
        self,
        id: PredictiveQueryID,
        project_id: Annotated[str, Query()],
    ) -> PQueryResponse:
        """
        Get a predictive query by ID.

        Args:
            id (str): The ID of the predictive query.

        Returns:
            IDResponse: The response object containing the ID of the predictive
            query.
        """
        with wrap_driver_errors(
            {PredictiveQueryDoesNotExistError: HTTPStatus.NOT_FOUND}):
            pq = await self._driver.get_predictive_query_v2(
                id, ProjectID(project_id))
            return PQueryResponse(
                name=pq.name,
                query_string=pq.query_string,
                graph=pq.graph,
            )

    async def get_pquery_metadata_v2(
        self,
        id: PredictiveQueryID,
        project_id: Annotated[str, Query()],
    ) -> PQueryMetadataResponse:
        """
        Get the task type and query type for a predictive query.

        Args:
            id (str): The ID of the predictive query.

        Returns:
            PQueryMetadataResponse: The task type and query type parsed from
            the predictive query.
        """
        with wrap_driver_errors(
            {PredictiveQueryDoesNotExistError: HTTPStatus.NOT_FOUND}):
            task_type, query_type = await self._driver.get_pquery_metadata(
                id, ProjectID(project_id))
            return PQueryMetadataResponse(
                task_type=task_type,
                query_type=query_type,
            )

    async def generate_holdout_presigned_url_v2(
            self,
            id: PredictiveQueryID,
            project_id: Annotated[str, Query()],
            ctx: RequestContext = Depends(get_ctx),
    ) -> OutputPresignedUrlResponse:
        """
        Generate a presigned URL for the predictive query's holdout dataset.

        Resolves the predictive query to its current training job and returns
        a download URL for that job's holdout data.
        """
        with wrap_driver_errors(
            {PredictiveQueryDoesNotExistError: HTTPStatus.NOT_FOUND}):
            training_job_id = await self._driver.get_current_training_job_id(
                ctx, id, ProjectID(project_id))
            if training_job_id is None:
                raise HTTPException(
                    HTTPStatus.BAD_REQUEST,
                    "Predictive Query has not been trained yet")
            training_job = await self._driver.get_training_job(
                ctx, training_job_id, strip_expensive_fields=True)
            if (training_job is None
                    or str(training_job.project_id) != project_id):
                raise TrainingJobDoesNotExistError(training_job_id)
            return await self._driver.generate_holdout_presigned_url(
                training_job_id)

    async def get_encoders_for_pquery_v2(
        self,
        id: PredictiveQueryID,
        project_id: Annotated[str, Query()],
        ctx: RequestContext = Depends(get_ctx),
    ) -> Optional[PQueryEncodersResponse]:
        """
        Get the encoders used for a predictive query.

        Returns a mapping from fully qualified table name to encoder
        representation, or null when the query has no completed training job.
        """
        with wrap_driver_errors(
            {PredictiveQueryDoesNotExistError: HTTPStatus.NOT_FOUND}):
            encoders = await self._driver.get_encoders_for_pquery(
                ctx, id, ProjectID(project_id))
            if encoders is None:
                return None
            return PQueryEncodersResponse(encoders=encoders)

    async def get_many_pqueries_v2(
        self,
        job_id_to_pq_fpid: Dict[Union[
            TrainingJobID,
            BatchPredictionJobID,
            GeneratePredictionTableJobID,
            GenerateTrainTableJobID,
        ], PredictiveQueryID],
        project_id: Annotated[str, Query()],
    ):
        """
        Get predictive query string and query task type.

        Args:
            A dictionary with job ID as a key and predictive query ID
            as value.

        Returns:
            A dictionary with job ID as a key and a tuple of query
            string and task type as value.
        """
        with wrap_driver_errors():
            # Skip validation since it is a read-only operation
            return await self._driver.get_many_predictive_queries_v2(
                job_id_to_pq_fpid,
                ProjectID(project_id),
                skip_validation=True,
            )

    async def get_default_model_plan(
        self,
        request: plan_api.SuggestModelPlanRequest,
        project_id: Annotated[str, Query()],
        # Internal testing only.
        dialect: Annotated[
            QueryDialect,
            Query(include_in_schema=False)] = QueryDialect.PQLANG_V2,
    ) -> ModelPlanResponse:
        """
        Suggest a model plan for a given query.

        Args:
            request (SuggestModelPlanRequest): The request object
                containing the query string, graph definition, and
                run mode. Although graph id is available,
                this method only uses graph def.
            dialect (QueryDialect, optional): The query dialect to use.
                Defaults to QueryDialect.PQLANG_V2.

        Returns:
            ModelPlanResponse: The response object containing the suggested
                model plan information.
        """
        if not request.graph_definition:
            raise ValueError("Graph definition must be set \
               to generate model planner.")
        with wrap_driver_errors(
            {PredictiveQueryDoesNotExistError: HTTPStatus.NOT_FOUND}):
            plan_info = (await
                         self._driver.get_pquery_model_plan_with_graph_def(
                             request.query_string,
                             request.graph_definition,
                             request.run_mode,
                             project_id=ProjectID(project_id),
                         ))
            return ModelPlanResponse(
                suggested_model_plan=plan_info,
                suggested_model_plan_yaml=model_plan_to_yaml(
                    plan_info.model_plan,
                    plan_info.task_type,
                    plan_info.query_type,
                ),
            )

    async def validate_pquery(
        self,
        request: PQueryValidationRequest,
        project_id: Annotated[str, Query()],
    ) -> jobs_api.ErrorDetails:
        """
            Validate a predictive query given PQueryValidationRequest. If
            the request contains model_plan, then the query is also validated
            in the context of the model_plan, otherwise simply in the context
            of the graph.

            Args:
                request (PQueryValidationRequest): The request object
                    containing the query details.
            Returns:
                ErrorDetails: The response object containing the validation
                    response.
            """
        # We do not support older versions of PQL in v1 APIs.
        # `validate` is a read-only check, so `check_unique_id=False` keeps
        # this route consistent with the SDK contract (validation must work
        # for unsaved queries that have no `PredictiveQueryID` yet).
        with wrap_driver_errors({KeyError: HTTPStatus.BAD_REQUEST}):
            response = await self._driver.validate_predictive_query_v2(
                request,
                project_id=ProjectID(project_id),
                check_unique_id=False,
            )
            return self._driver.convert_validation_response_to_error_details(
                response)

    async def get_allowed_bp_query_overrides_v2(
            self,
            training_job_id: TrainingJobID,
            request_context: RequestContext = Depends(get_ctx),
    ) -> EditableQuery:
        """
        Get allowed batch prediction query overrides for a training job.

        Args:
            training_job_id (TrainingJobID): The ID of the training job.

        Returns:
            EditableQuery: The editable query object containing allowed
            overrides.
        """
        with wrap_driver_errors(
            {TrainingJobDoesNotExistError: HTTPStatus.NOT_FOUND}):
            return await self._driver.get_allowed_bp_query_overrides_v2(
                request_context,
                training_job_id,
            )

    async def insert_filter_overrides(
            self,
            training_job_id: TrainingJobID,
            filter_overrides: InsertFilterOverridesRequest,
            request_context: RequestContext = Depends(get_ctx),
    ) -> EditableQuery:
        """
        V2 API for inserting filter overrides for a training job.

        Args:
            training_job_id (TrainingJobID): The ID of the training job.
            filter_overrides (EditableQuery): The filter overrides to insert.
            request_context (RequestContext): The request context.
        """
        with wrap_driver_errors({KeyError: HTTPStatus.BAD_REQUEST}):
            return await self._driver.insert_filter_overrides_v2(
                request_context,
                training_job_id,
                filter_overrides,
            )
