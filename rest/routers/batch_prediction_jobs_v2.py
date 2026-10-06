# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from http import HTTPStatus
from logging import getLogger
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from kumo_core.explain.schemas import EntityOutput, HistoricalItems
from kumo_core.types.schema import ColumnArgs
from kumoapi.explain import Subgraph
from kumoapi.pquery import QueryType
from pydantic import ValidationError

from kumo.config.explain import StrategyMappings
from kumo.connector.connector import ConnectorError
from kumo.pquery.batch_pred_table_generator import PredictionTableGeneratorV2
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.batch_prediction_job import (
    BatchPredictionJobAnchorTime,
    BatchPredictionJobResponse,
    BatchPredictionJobStats,
    get_bp_columns_by_task_type,
)
from rest.config.batch_prediction_job_v2 import (
    BatchPredictionInternalRequest,
    PredictionTableSchemaRequest,
    PredictionTableSchemaSampleColumn,
    PredictionTableSchemaSamplesRequest,
)
from rest.config.evaluation import XAIResponse
from rest.config.predictive_query import OutputPresignedUrlResponse
from rest.config.status import IDResponse, RestStatusCode
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.jobs_driver import JobsDriver
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.driver.rest_api_driver import RestApiDriver
from rest.routers import DISABLE_UI_WARNINGS
from rest.utils.project_scope import (
    require_project_id,
    require_requested_project_visible_for_user,
    require_visible_project_resource_for_user,
)
from rest.utils.routers import (
    PydanticValidationHTTPException,
    ValidationHTTPException,
    wrap_validation_exceptions,
)
from statedb.config.common import (
    BatchPredictionJobID,
    ErrWarnArgs,
    ErrWarnResponse,
    GenerateTrainTableJobID,
    JobSource,
    PredictiveQueryID,
    ProjectID,
)
from statedb.utils.errors import (
    BatchPredictionDoesNotExistError,
    PredictiveQueryDoesNotExistError,
)

log = getLogger(__name__)


class BatchPredictionV1Resource:
    def __init__(
        self,
        auth: Authentication,
        jobs_driver: JobsDriver,
        data_driver: RestApiDriver,
        ml_job_backend: MLJobBackend,
    ) -> None:
        self._auth = auth
        self._driver: JobsDriver = jobs_driver
        self._data_driver = data_driver
        self._pquery_driver: PQueryApiDriver = data_driver.pquery_driver
        self._ml_job_backend = ml_job_backend
        self.router = APIRouter(
            prefix="/v1/batch_prediction_jobs",
            tags=["API v1 (Under Development): BatchPredictionJob"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={HTTPStatus.NOT_FOUND: {
                "description": "Not found"
            }},
        )

        self.router.add_api_route(
            "/",
            self.create_batch_prediction_job,
            response_model=IDResponse,
            methods=['POST'],
        )

        self.router.add_api_route(
            "/output_schema",
            self.get_prediction_table_schema,
            response_model=List[ColumnArgs],
            methods=['POST'],
        )

        self.router.add_api_route(
            "/{id}/prediction_table_schema",
            self.get_prediction_table_schema_by_pquery,
            response_model=List[ColumnArgs],
            methods=['GET'],
        )

        self.router.add_api_route(
            "/samples_for_output_schema",
            self.suggest_sample_predictive_table_schema,
            response_model=List[PredictionTableSchemaSampleColumn],
            methods=['POST'],
        )

        self.router.add_api_route(
            "/{prediction_job_id}/sdk",
            self.prediction_to_sdk,
            response_model=str,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}",
            self.get_batch_prediction_job,
            response_model=BatchPredictionJobResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/cancel",
            self.cancel,
            methods=['POST'],
            description="Cancel a batch prediction job.",
            response_model=bool,
        )

        self.router.add_api_route(
            "/{id}/prediction_anchor_time",
            self.get_prediction_anchor_time,
            response_model=BatchPredictionJobAnchorTime,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/prediction_stats",
            self.get_prediction_stats,
            response_model=BatchPredictionJobStats,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/download_bp",
            self.generate_bp_presigned_url,
            response_model=OutputPresignedUrlResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/get_warnings",
            self.get_warnings,
            response_model=Dict[str, ErrWarnResponse],
            methods=['POST'],
        )

        self.router.add_api_route(
            "/{id}/xai",
            self.get_xai,
            response_model=XAIResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/available_xai_entities",
            self.get_available_xai_entities,
            response_model=StrategyMappings,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/subgraph/{entity_id}",
            self.get_entity_subgraph,
            response_model=Subgraph,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/entity_output/{entity_id}",
            self.get_entity_output,
            response_model=EntityOutput,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{id}/entity_historical_items/{entity_id}",
            self.get_entity_historical_items,
            response_model=HistoricalItems,
            methods=['GET'],
        )

    def _authorize_batch_prediction_read(
        self,
        response: Optional[BatchPredictionJobResponse],
        project_id: str,
        ctx: RequestContext,
    ) -> BatchPredictionJobResponse:
        return require_visible_project_resource_for_user(
            response,
            requested_project_id=project_id,
            user_authorization=ctx.user_authorization,
            not_found_error_factory=lambda: HTTPException(
                HTTPStatus.NOT_FOUND,
                RestStatusCode.PREDICTION_JOB_DOES_NOT_EXIST.value,
            ),
        )

    async def cancel(
        self, id: BatchPredictionJobID, project_id: str,
        ctx: RequestContext = Depends(get_ctx)) -> bool:
        project_id = require_project_id(project_id)
        require_requested_project_visible_for_user(
            project_id,
            user_authorization=ctx.user_authorization,
            not_found_error_factory=lambda: HTTPException(
                HTTPStatus.NOT_FOUND,
                RestStatusCode.PREDICTION_JOB_DOES_NOT_EXIST.value,
            ),
        )
        response = await self._pquery_driver.get_batch_prediction_job(ctx, id)
        self._authorize_batch_prediction_read(response, project_id, ctx)
        log.info('Send cancellation request for job: %s', id)
        return await self._ml_job_backend.cancel_batch_prediction_job(id)

    async def get_warnings(
            self,
            args: ErrWarnArgs,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> Dict[str, ErrWarnResponse]:
        """Get all warnings and errors after validating the given Batch
           Prediction Job IDs.

        Args:
            A list of BP identifier strings for which errors and warnings
            if any will be returned.

        Returns:
            Dict[ID, ErrWarnResponse] containing errors and warnings.
        """
        if DISABLE_UI_WARNINGS:
            return {
                BatchPredictionJobID(id):
                ErrWarnResponse(errors=[], warnings=[], info_items=[])
                for id in args.ids
            }

        require_requested_project_visible_for_user(
            project_id,
            user_authorization=ctx.user_authorization,
            not_found_error_factory=lambda: HTTPException(
                HTTPStatus.NOT_FOUND,
                RestStatusCode.PREDICTION_JOB_DOES_NOT_EXIST.value,
            ),
        )
        for bp_id in args.ids:
            response = await self._pquery_driver.get_batch_prediction_job(
                ctx, BatchPredictionJobID(bp_id))
            self._authorize_batch_prediction_read(response, project_id, ctx)

        with wrap_validation_exceptions(
                not_exists_error=BatchPredictionDoesNotExistError):
            return await self._pquery_driver.get_batch_prediction_warnings(
                [BatchPredictionJobID(id) for id in args.ids], ctx)

    async def get_batch_prediction_job(
            self,
            id: BatchPredictionJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> BatchPredictionJobResponse:
        r"""Get the config, progress, and/or results for a batch prediction
        job.

        Args:
            id (str): Id of the batch prediction job to get

        Returns:
            - id (str): the globally unique identifier for the object
            - config (object): fields related to the batch prediction job
              configuration at launch time.
            - status (string): NOT_STARTED|PROCESSING|DONE|FAILED
            - progress (object): only present if status=PROCESSING. Contains
              information about the in-progress execution, such as estimated
              time remaining.
            - results (object): only present if status=DONE. Contains
              metrics about the batch prediction output files
            - timestamps (ResourceTimestamp): object creation and updated
              timestamps

        Errors:
            HTTPStatus.NOT_FOUND: If the batch prediction job does not exist.
        """
        require_requested_project_visible_for_user(
            project_id,
            user_authorization=ctx.user_authorization,
            not_found_error_factory=lambda: HTTPException(
                HTTPStatus.NOT_FOUND,
                RestStatusCode.PREDICTION_JOB_DOES_NOT_EXIST.value,
            ),
        )
        response = await self._pquery_driver.get_batch_prediction_job(ctx, id)
        response = self._authorize_batch_prediction_read(
            response, project_id, ctx)
        response.download_supported = (
            await self._pquery_driver.supports_batch_prediction_download(id))
        return response

    async def get_prediction_anchor_time(
        self,
        id: BatchPredictionJobID,
        project_id: str,
        ctx: RequestContext = Depends(get_ctx),
    ) -> BatchPredictionJobAnchorTime:
        r"""Get the prediction anchor time used for this job

        Args:
            id (str): Id of the batch prediction job to get anchor time for

        Returns:
            - None if the job is not done yet
            - datetime if we have the anchor time

        Errors:
            HTTPStatus.NOT_FOUND: If the batch prediction job does not exist.
        """
        require_requested_project_visible_for_user(
            project_id,
            user_authorization=ctx.user_authorization,
            not_found_error_factory=lambda: HTTPException(
                HTTPStatus.NOT_FOUND,
                RestStatusCode.PREDICTION_JOB_DOES_NOT_EXIST.value,
            ),
        )
        response = await self._pquery_driver.get_batch_prediction_job(ctx, id)
        response = self._authorize_batch_prediction_read(
            response, project_id, ctx)
        if response.config.prediction_time is not None:
            return BatchPredictionJobAnchorTime(
                anchor_time=response.config.prediction_time)

        # First try to load from the job directory
        home_dir = await self._ml_job_backend.get_job_home_dir(id)
        anchor_time = await (
            self._pquery_driver.get_prediction_anchor_time_from_per_job_dir(
                id,
                home_dir,
            ))
        if anchor_time is not None:
            return anchor_time

        # If not found in job directory, try loading from cache
        return await self._pquery_driver.get_prediction_anchor_time(id, ctx)

    async def get_prediction_stats(
            self,
            id: BatchPredictionJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> BatchPredictionJobStats:
        """Returns the col stats for a batch prediction job if
        available.
        """
        response = await self._pquery_driver.get_batch_prediction_job(ctx, id)
        self._authorize_batch_prediction_read(response, project_id, ctx)
        with wrap_validation_exceptions(BatchPredictionDoesNotExistError):
            try:
                return await self._pquery_driver.get_prediction_stats(id, ctx)
            except ConnectorError as e:
                log.exception(e)
                raise HTTPException(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    'This query is too old and does not have any '
                    'statistics available',
                )

    async def generate_bp_presigned_url(
            self,
            id: BatchPredictionJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> OutputPresignedUrlResponse:
        r"""Generate the batch prediction download url.

        Args:
            id (str): ID of the Predictive query to download the bp data.

        Returns:
            OutputPresignedUrlResponse class which has the presigned url.

        Raises:
            - ValueError: if the query does not exists, the bp data is too
                          big, the data does not exist, or data expired.
        """
        try:
            require_requested_project_visible_for_user(
                project_id,
                user_authorization=ctx.user_authorization,
                not_found_error_factory=lambda: HTTPException(
                    HTTPStatus.NOT_FOUND,
                    RestStatusCode.PREDICTION_JOB_DOES_NOT_EXIST.value,
                ),
            )
            bp_job_response = await self._pquery_driver.get_batch_prediction_job(
                ctx, id)
            bp_job_response = self._authorize_batch_prediction_read(
                bp_job_response, project_id, ctx)
            batch_prediction_job_id = bp_job_response.id
            response = await self._pquery_driver.generate_bp_presigned_url(
                batch_prediction_job_id)
            return response
        except (ValueError, KeyError, NotImplementedError) as err:
            raise ValidationHTTPException(str(err), err)

    async def get_xai(
            self,
            id: BatchPredictionJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> XAIResponse:
        r"""Get explanation for a batch prediction job.
        Args:
            id (str): Id for the batch prediction job.

        Returns (XAIResponse): contain column-level contribution scores and
            column analysis, as well as navigation tree starting from root
            (entity) table.
        """
        with wrap_validation_exceptions(BatchPredictionDoesNotExistError):
            job = await self._pquery_driver.get_batch_prediction_job(ctx, id)
            job = self._authorize_batch_prediction_read(job, project_id, ctx)
            pq_id = job.config.predictive_query_id
            return await self._data_driver.get_bp_xai(ctx, pq_id, id)

    async def get_available_xai_entities(
            self,
            id: BatchPredictionJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> StrategyMappings:
        r"""Get preselection strategy mappings of prediction-level XAI
        for a batch prediction.

        Args:
            id (str): Id of the trained predictive query to get eval stats for.

        Returns (StrategyMappings): Strategy mappings which contain the
            mappings of preselection strategy and entities.
        """
        with wrap_validation_exceptions(BatchPredictionDoesNotExistError):
            job = await self._pquery_driver.get_batch_prediction_job(ctx, id)
            self._authorize_batch_prediction_read(job, project_id, ctx)
            return await self._data_driver.get_bp_available_xai_entities(
                ctx, id)

    async def get_entity_subgraph(
            self,
            id: BatchPredictionJobID,
            entity_id: str,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> Subgraph:
        r"""Get XAI subgraph of an entity.

        Args:
            id (str): Id of the batch prediction to get stats for.
            entity_id (str): Id of the entity to get subgraph for.

        Returns (Subgraph): Subgraph of the entity.
        """
        with wrap_validation_exceptions(BatchPredictionDoesNotExistError):
            job = await self._pquery_driver.get_batch_prediction_job(ctx, id)
            self._authorize_batch_prediction_read(job, project_id, ctx)
            return await self._data_driver.get_bp_subgraph(ctx, id, entity_id)

    async def get_entity_output(
            self,
            id: BatchPredictionJobID,
            entity_id: str,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> EntityOutput:
        r"""Get XAI output of an entity.

        Args:
            id (str): Id of the batch prediction to get stats for.
            entity_id (str): Id of the entity to get subgraph for.

        Returns (EntityOutput): Output of the entity.
        """
        with wrap_validation_exceptions(BatchPredictionDoesNotExistError):
            job = await self._pquery_driver.get_batch_prediction_job(ctx, id)
            self._authorize_batch_prediction_read(job, project_id, ctx)
            return await self._data_driver.get_bp_entity_output(
                ctx, id, entity_id)

    async def get_entity_historical_items(
            self,
            id: BatchPredictionJobID,
            entity_id: str,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> HistoricalItems:
        r"""Get XAI historical items of an entity for link prediction tasks.

        Args:
            id (str): Id of the batch prediction to get stats for.
            entity_id (str): Id of the entity to get subgraph for.

        Returns (HistoricalItems): Historical items of the entity.
        """
        with wrap_validation_exceptions(BatchPredictionDoesNotExistError):
            job = await self._pquery_driver.get_batch_prediction_job(ctx, id)
            self._authorize_batch_prediction_read(job, project_id, ctx)
            return await self._data_driver.get_bp_entity_historical_items(
                ctx, id, entity_id)

    async def create_batch_prediction_job(
            self,
            request: BatchPredictionInternalRequest,
            ctx: RequestContext = Depends(get_ctx),
    ) -> IDResponse:
        """
        Creates a batch prediction job in V1 APIs.
        Args:
            request (BatchPredictionRequest): The batch prediction request
            object.
            ctx (RequestContext, optional): The request context, defaults to
            the context provided by Depends(get_ctx).
        Returns:
            str: The ID of the created batch prediction job.
        Raises:
            HTTPException: If there are validation errors in the response, an
            HTTP 500 error is raised with the error message.
        """
        require_project_id(request.project_id)
        try:
            response = await self._driver.create_batch_prediction_job(
                ctx,
                request,
                # Currently this API is only used by the UI
                JobSource.UI,
            )
            return response
        except ValidationError as err:
            raise PydanticValidationHTTPException(err)
        except ValueError as err:
            raise ValidationHTTPException(str(err), err)

    async def get_prediction_table_schema(
            self,
            request: PredictionTableSchemaRequest,
            ctx: RequestContext = Depends(get_ctx),
    ) -> List[ColumnArgs]:
        return get_bp_columns_by_task_type(
            task_type=request.task_type,
            is_temporal=request.query_type == QueryType.TEMPORAL,
            entity_col_name=PredictionTableGeneratorV2.ENTITY,
        )

    async def get_prediction_table_schema_by_pquery(
            self,
            id: PredictiveQueryID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> List[ColumnArgs]:
        """Prediction-table output schema resolved from a predictive query.

        Uses the query's actual entity column name (unlike
        `get_prediction_table_schema`, which uses the generic placeholder).
        Replaces the v0 `batch_prediction_workflows` schema endpoint.
        """
        with wrap_validation_exceptions(PredictiveQueryDoesNotExistError):
            require_requested_project_visible_for_user(
                project_id,
                user_authorization=ctx.user_authorization,
                not_found_error_factory=lambda: HTTPException(
                    HTTPStatus.NOT_FOUND,
                    RestStatusCode.PREDICTIVE_QUERY_DOES_NOT_EXIST.value,
                ),
            )
            # ``get_pquery_metadata`` resolves both the v1 (name) and v2
            # (public fpid) id forms, unlike ``get_predictive_query`` which
            # asserts the id carries no fpid.
            (task_type, query_type), pq_object = await asyncio.gather(
                self._pquery_driver.get_pquery_metadata(id,
                                                        ProjectID(project_id)),
                self._pquery_driver.get_predictive_query_object(
                    id, ProjectID(project_id),
                    self._pquery_driver._pquery_settings),
            )
            if pq_object is None:
                raise PredictiveQueryDoesNotExistError(id)
            entity_col_name = pq_object.get_bp_entity_col_name()
            return get_bp_columns_by_task_type(
                task_type=task_type,
                is_temporal=query_type == QueryType.TEMPORAL,
                entity_col_name=entity_col_name,
            )

    async def suggest_sample_predictive_table_schema(
        self,
        request: PredictionTableSchemaSamplesRequest,
        ctx: RequestContext = Depends(get_ctx),
    ) -> List[PredictionTableSchemaSampleColumn]:
        """
        Complimentary endpoint to `get_prediction_table_schema`.
        Fetches samples from the training table to show a preview
        of the prediction job.

        Args:
            PredictionTableSchemaSamplesRequest
                - task type
                - training table job ID
        Response:
            An array `SampleColumn`s. Each sample column
            has a name and an array of 10 samples.
            The exact contents of the response depend on the
            task type.
        """
        if not GenerateTrainTableJobID.is_valid_id(request.train_table_job_id):
            raise ValueError(f'{request.train_table_job_id} is not a ' +
                             'valid train table job ID.')
        train_table_job_id = GenerateTrainTableJobID(request.train_table_job_id)
        return await self._driver.get_training_job_samples(
            train_table_job_id, request.task_type)

    async def prediction_to_sdk(
            self,
            prediction_job_id: BatchPredictionJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> str:
        r"""
        Used to generate a fully executable Python SDK script to reproduce a prediction pipeline
        in the SDK. The script contains:
            1. Connector(s) config(s). If there are tables from multiple connectors,
            we load the config for each connector.
            2. Table configs structured as Python dictionaries.
            3. Graph edge configs.
            4. Using the aforementioned configs to setup a graph in the SDK.
            5. Predictive query config.
            6. Training table config.
            7. Model plan config.
            8. Use the graph and configs to generate a training table and training jobs.
               Additionally, give the option to load a training table with ID
               instead of generating a new one.
            9. Separate connector and/or predictive query config for a prediction job.
            10. Prediction and prediction table configs.

        This method is part of the UI to SDK feature and will be called in the UI.

        Args:
            id (BatchPredictionJobID): ID of the prediction job.
            ctx (RequestContext): Request context.
        Response:
            str: The complete python script for an SDK prediction workflow as a string.

        Raises:
            HTTPException: If prediction job cannot be retrieved.
        """
        with wrap_validation_exceptions(BatchPredictionDoesNotExistError):
            job = await self._pquery_driver.get_batch_prediction_job(
                ctx, prediction_job_id)
            self._authorize_batch_prediction_read(job, project_id, ctx)
            return await self._driver.prediction_to_sdk(prediction_job_id, ctx)
