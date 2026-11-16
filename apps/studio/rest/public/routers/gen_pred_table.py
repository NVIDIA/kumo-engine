# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from http import HTTPStatus
from typing import Callable, List, Optional

import kumoapi.jobs as api
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.params import Query
from typing_extensions import Annotated

from io_utils import logging
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.batch_prediction_job import BatchPredictionJobAnchorTime
from rest.config.status import EmptyResponse, IDResponse, PatchResponse
from rest.deps.authentication import (
    Authentication,
    RequestContext,
    get_ctx,
    visible_project_ids_or_none,
)
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.public.routers.common import (
    list_job_tags,
    require_project_id_valid,
    resolve_requested_project,
    update_job_tags,
    user_can_access_project,
    validate_custom_job_tags,
)
from rest.utils.public import (
    presign_parquet_table_data,
    wrap_driver_errors,
    wrap_rpc_error,
)
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    JobSource,
    JobType,
    PredictiveQueryID,
    ProjectID,
)

log = logging.getLogger(__name__)


class GeneratePredictionTableRouter:
    def __init__(
        self,
        api_key_auth: Callable,
        driver: PQueryApiDriver,
        auth: Authentication,
        ml_job_backend: MLJobBackend,
    ) -> None:
        self._driver = driver
        self._auth = auth

        self._ml_job_backend = ml_job_backend

        self.router = APIRouter(
            prefix='/genpredtable_jobs',
            tags=["API v1 (Under Development): Generate-Prediction-Table Jobs"],
            dependencies=[Depends(api_key_auth)],
        )

        self.router.add_api_route(
            '',
            self.create,
            response_model=IDResponse,
            methods=['POST'],
            summary='Create a Generate-Prediction-Table job.',
            response_description='ID of the genereate-prediction-table job.',
        )

        self.router.add_api_route(
            '/{id}',
            self.get,
            response_model=api.GeneratePredictionTableJobResource,
            methods=['GET'],
            summary='Get a Generate-Prediction-Table job.',
            response_description='The genereate-prediction-table job resource',
        )

        self.router.add_api_route(
            '',
            self.list,
            response_model=List[api.GeneratePredictionTableJobResource],
            methods=['GET'],
            summary='List Generate-Prediction-Table jobs.',
            response_description='List of job resources',
        )

        self.router.add_api_route(
            '/{id}/table_data',
            self.get_table_data,
            methods=['GET'],
            summary=('Get prediction table data URL for a successfully '
                     'completed Generate-Prediction-Table job.'),
            response_description=(
                'The table data URL(s) for a successfully genereated '
                'prediction table data (parquet file); could be a list of '
                'multiple URLs if the generated table partitioned into '
                'multiple parts.'),
        )

        self.router.add_api_route(
            '/{id}/get_anchor_time',
            self.get_anchor_time,
            methods=['GET'],
            summary=("Returns the anchor time of a job to generate a "
                     "batch prediction table, or None if not present."),
        )

        self.router.add_api_route(
            "/{id}/cancel",
            self.cancel,
            response_model=EmptyResponse,
            methods=['POST'],
            summary='Cancel a Generate-Prediction-Table job if it is running.',
            response_description='Whether the job is successfully canceled.',
        )

        self.router.add_api_route(
            '/{id}/tags',
            self.update_tags,
            methods=['PATCH'],
            response_model=PatchResponse,
            summary='Update(add/mutate/remove) job tags.',
        )

        # Errors:
        self.router.add_api_route(
            '/{id}/get_errors',
            self.get_errors,
            methods=['GET'],
            response_model=api.ErrorDetails,
        )

    async def get_errors(
            self,
            id: GeneratePredictionTableJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> api.ErrorDetails:
        await self.get(id, ctx)  # verifies existence and project access
        return await self._driver.get_errors_for_job(id)

    async def create(
            self,
            request: api.GeneratePredictionTableRequest,
            project_id: Annotated[str, Query()],
            ctx: RequestContext = Depends(get_ctx),
    ) -> IDResponse:
        validate_custom_job_tags(request.job_tags)
        require_project_id_valid(ctx, project_id)
        with wrap_driver_errors():
            job_id = await self._driver.start_generate_pred_table(
                request,
                job_source=JobSource.SDK,
                is_child_job=False,
                project_id=ProjectID(project_id),
            )
            return IDResponse(job_id)

    async def get(
        self,
        id: GeneratePredictionTableJobID,
        ctx: RequestContext = Depends(get_ctx),
    ) -> api.GeneratePredictionTableJobResource:
        with wrap_rpc_error():
            jobs = await self._ml_job_backend.get_generate_pred_table_jobs(
                {id},
                project_ids=visible_project_ids_or_none(ctx.user_authorization),
            )
            if not jobs:
                raise HTTPException(
                    HTTPStatus.NOT_FOUND,
                    f'Generate prediction table job {id} is not found.')
            job = jobs[0]
            if not user_can_access_project(ctx, job.project_id):
                raise HTTPException(
                    HTTPStatus.NOT_FOUND,
                    f'Generate prediction table job {id} is not found.',
                )
            tags = await self._ml_job_backend.get_job_tags_if_exists(
                JobType.GENERATE_PREDICTION_TABLE_JOB, id)
            job.tags = tags.tags if tags is not None else {}
            return job

    async def get_table_data(
            self,
            id: GeneratePredictionTableJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> List[str]:
        job = await self.get(id, ctx)
        if job.job_status_report.status != api.JobStatus.DONE:
            raise HTTPException(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                f'Unable to get prediction table data URL for job {id} '
                f'in status: {job.job_status_report.status}')

        with wrap_rpc_error():
            result = await self._ml_job_backend.get_pred_table_result(id)

        return await presign_parquet_table_data.coroutine(result.pred_table_path
                                                          )

    async def get_anchor_time(
            self,
            id: GeneratePredictionTableJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> Optional[datetime]:
        job = await self.get(id, ctx)
        if job.job_status_report.status != api.JobStatus.DONE:
            raise HTTPException(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Cannot return the anchor time of a prediction table "
                "before the prediction table has finished generation.")

        # NOTE this conversion is OK since the ID of a batch prediction job
        # that has prediction table generation run in standalone mode is the
        # the same as that job's ID:
        bp_id = BatchPredictionJobID(id)

        home_dir = await self._ml_job_backend.get_job_home_dir(bp_id)
        anchor_time = await (
            self._driver.get_prediction_anchor_time_from_per_job_dir(
                bp_id,
                home_dir,
            ))
        if anchor_time is not None:
            return anchor_time.anchor_time

        out = BatchPredictionJobAnchorTime()
        await out.load_from_cache.coroutine(bp_id)
        return out.anchor_time

    async def list(
        self,
        request: Request,
        response: Response,
        pquery_name: Optional[PredictiveQueryID] = None,
        pquery_id: Optional[PredictiveQueryID] = None,
        job_status: Optional[api.JobStatus] = None,
        limit: Annotated[Optional[int], Query(ge=1)] = None,
        project_id: Annotated[Optional[str], Query()] = None,
        ctx: RequestContext = Depends(get_ctx),
    ) -> List[api.GeneratePredictionTableJobResource]:
        requested_project, project_scope_matches = await resolve_requested_project(
            ctx, self._driver, project_id)
        job_tags = await list_job_tags(JobType.GENERATE_PREDICTION_TABLE_JOB,
                                       request, self._driver, response)
        if not project_scope_matches:
            return []
        job_ids = set(GeneratePredictionTableJobID(id) for id in job_tags)
        project_ids = ({requested_project} if requested_project is not None else
                       visible_project_ids_or_none(ctx.user_authorization))
        with wrap_rpc_error():
            jobs = await self._ml_job_backend.get_generate_pred_table_jobs(
                job_ids,
                project_ids=project_ids,
                status=job_status,
                limit=limit,
            )

        for job in jobs:
            job.tags = job_tags[job.job_id].tags

        jobs.reverse()  # We need reverse chronological order.
        return jobs

    async def cancel(
            self,
            id: GeneratePredictionTableJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> EmptyResponse:
        await self.get(id, ctx)  # verifies existence and project access
        log.info('Send cancellation request for job: %s', id)
        with wrap_rpc_error():
            await self._ml_job_backend.cancel_generate_pred_table_job(id)
        return EmptyResponse()

    async def update_tags(
            self,
            id: GeneratePredictionTableJobID,
            request: Request,
            ctx: RequestContext = Depends(get_ctx),
    ) -> PatchResponse:
        job = await self.get(id, ctx)  # verifies existence and project access
        with wrap_driver_errors():
            return await update_job_tags(JobType.GENERATE_PREDICTION_TABLE_JOB,
                                         id, job.project_id, request,
                                         self._driver)
