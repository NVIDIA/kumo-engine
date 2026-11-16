# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from http import HTTPStatus
from typing import Callable, Dict, List, Optional

import kumoapi.jobs as api
from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
)
from kumo_core.model_plan import ModelPlanValueError
from kumoapi.jobs import TrainingTableSpec
from kumoapi.source_table import SourceTableType
from kumoapi.train import TrainingStage
from typing_extensions import Annotated

from io_utils import fsutil, logging
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
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
    GenerateTrainTableJobID,
    JobSource,
    JobType,
    PredictiveQueryID,
    ProjectID,
)
from statedb.utils.errors import PredictiveQueryDoesNotExistError

log = logging.getLogger(__name__)


class GenerateTrainTableRouter:
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

        self.router_v1 = APIRouter(
            prefix='/gentraintable_jobs',
            tags=["API v1 (Under Development): Generate-Train-Table Jobs"],
            dependencies=[Depends(api_key_auth)],
        )

        self.router_v1.add_api_route(
            '',
            self.create,
            response_model=IDResponse,
            methods=['POST'],
            summary='Create a Generate-Train-Table job.',
            response_description='ID of the genereate-train-table job.',
        )

        self.router_v1.add_api_route(
            '/{id}',
            self.get,
            response_model=api.GenerateTrainTableJobResource,
            methods=['GET'],
            summary='Get a Generate-Train-Table job.',
            response_description='The genereate-train-table job resource.',
        )
        self.router_v1.add_api_route(
            '',
            self.list,
            response_model=List[api.GenerateTrainTableJobResource],
            methods=['GET'],
            summary='List Generate-Train-Table jobs.',
            response_description='List of job resources',
        )

        self.router_v1.add_api_route(
            '/{id}/table_data',
            self.get_table_data,
            methods=['GET'],
            summary=('Get table data URL for a successfully completed '
                     'Generate-Train-Table job.'),
            response_description=(
                'The data URL for a successfully genereated training table.'),
        )

        self.router_v1.add_api_route(
            '/{id}/split_masks',
            self.get_split_masks,
            response_model=Dict[str, str],
            methods=['GET'],
            summary=(
                'Get URLs for train table split mask for train/val/test '
                'stages, produced by a successful Generate-Train-Table job.'),
            response_description=(
                'Dictionary of training table split mask file URLs, '
                'keyed by training stage (train, val, test)'),
        )

        self.router_v1.add_api_route(
            "/{id}/cancel",
            self.cancel,
            response_model=EmptyResponse,
            methods=['POST'],
            summary='Cancel a Generate-Train-Table job if it is running.',
            response_description='Whether the job is successfully canceled.',
        )

        self.router_v1.add_api_route(
            '/{id}/tags',
            self.update_tags,
            methods=['PATCH'],
            response_model=PatchResponse,
            summary='Update(add/mutate/remove) job tags.',
        )

        self.router_v1.add_api_route(
            '/{id}/progress',
            self.get_progress,
            methods=['GET'],
            response_model=Dict[str, int],
        )

        self.router_v1.add_api_route(
            '/{id}/get_errors',
            self.get_errors,
            methods=['GET'],
            response_model=api.ErrorDetails,
        )

        self.router_v1.add_api_route(
            '/{id}/validate_custom_train_table',
            self.validate_custom_train_table,
            methods=['POST'],
            response_model=api.ValidationResponse,
        )

    async def get_errors(
            self,
            id: GenerateTrainTableJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> api.ErrorDetails:
        await self.get(id, ctx)  # verifies existence and project access
        return await self._driver.get_errors_for_job(id)

    async def create(
            self,
            request: api.GenerateTrainTableRequest,
            project_id: Annotated[str, Query()],
            ctx: RequestContext = Depends(get_ctx),
    ) -> IDResponse:
        validate_custom_job_tags(request.job_tags)
        require_project_id_valid(ctx, project_id)
        with wrap_driver_errors({
                PredictiveQueryDoesNotExistError:
                HTTPStatus.UNPROCESSABLE_ENTITY,
                ModelPlanValueError:
                HTTPStatus.UNPROCESSABLE_ENTITY,
        }):
            job_id = await self._driver.start_generate_train_table(
                request,
                job_source=JobSource.SDK,
                project_id=ProjectID(project_id),
            )

        return IDResponse(job_id)

    async def get(
        self,
        id: GenerateTrainTableJobID,
        ctx: RequestContext = Depends(get_ctx),
    ) -> api.GenerateTrainTableJobResource:
        with wrap_rpc_error(
            {PredictiveQueryDoesNotExistError:
             HTTPStatus.UNPROCESSABLE_ENTITY}):
            jobs = await self._ml_job_backend.get_generate_train_table_jobs(
                {id},
                project_ids=visible_project_ids_or_none(ctx.user_authorization),
            )
            if not jobs:
                raise HTTPException(
                    HTTPStatus.NOT_FOUND,
                    f'Generate train table job {id} is not found.')
            tags = await self._ml_job_backend.get_job_tags_if_exists(
                JobType.GENERATE_TRAIN_TABLE_JOB, id)
            job = jobs[0]
            if not user_can_access_project(ctx, job.project_id):
                raise HTTPException(
                    HTTPStatus.FORBIDDEN,
                    f'Project {job.project_id} is not accessible.',
                )
            job.tags = tags.tags if tags is not None else {}
            return job

    async def get_progress(
            self,
            id: GenerateTrainTableJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> Dict[str, int]:
        _ = await self.get(id, ctx)
        out = await self._driver.get_train_table_generation_stats(id)
        if out is None:
            return dict()
        return dict(num_expected_iterations=out.num_expected_iterations,
                    num_finished_iterations=out.num_finished_iterations)

    async def get_table_data(
            self,
            id: GenerateTrainTableJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> List[str]:
        job = await self.get(id, ctx)
        if job.job_status_report.status != api.JobStatus.DONE:
            raise HTTPException(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                f'Unable to get training table data URL for job {id} '
                f'in status: {job.job_status_report.status}')
        with wrap_rpc_error():
            result = await self._ml_job_backend.get_train_table_result(id)

        return await presign_parquet_table_data.coroutine(
            result.train_table_path)

    async def get_split_masks(
            self,
            id: GenerateTrainTableJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> Dict[TrainingStage, str]:
        job = await self.get(id, ctx)
        if job.job_status_report.status != api.JobStatus.DONE:
            raise HTTPException(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                f'Unable to get training table split masks for job {id} '
                f'in status: {job.job_status_report.status}')
        with wrap_rpc_error():
            result = await self._ml_job_backend.get_train_table_result(id)

        mask_dir = result.train_table_materialization_path
        helper = fsutil.get_presigned_helper(mask_dir)
        mask_urls = {}
        for stage in TrainingStage:
            mask_path = os.path.join(mask_dir, f'{stage}_mask.pt')
            if fsutil.exists(mask_path):
                if helper is not None:
                    mask_urls[stage] = (
                        await helper.generate_download_url.coroutine(mask_path))
                else:
                    mask_urls[stage] = fsutil.customer_facing_path(mask_path)

        return mask_urls

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
    ) -> List[api.GenerateTrainTableJobResource]:
        requested_project, project_scope_matches = await resolve_requested_project(
            ctx, self._driver, project_id)
        job_tags = await list_job_tags(JobType.GENERATE_TRAIN_TABLE_JOB,
                                       request, self._driver, response)
        if not project_scope_matches:
            return []
        job_ids = set(GenerateTrainTableJobID(id) for id in job_tags)
        project_ids = ({requested_project} if requested_project is not None else
                       visible_project_ids_or_none(ctx.user_authorization))
        with wrap_rpc_error():
            jobs = await self._ml_job_backend.get_generate_train_table_jobs(
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
            id: GenerateTrainTableJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> EmptyResponse:
        await self.get(id, ctx)  # verifies existence and project access
        log.info('Send cancellation request for job: %s', id)
        with wrap_rpc_error():
            await self._ml_job_backend.cancel_generate_train_table_job(id)
        return EmptyResponse()

    async def update_tags(
            self,
            id: GenerateTrainTableJobID,
            request: Request,
            ctx: RequestContext = Depends(get_ctx),
    ) -> PatchResponse:
        job = await self.get(id, ctx)  # verifies existence and project access
        return await update_job_tags(JobType.GENERATE_TRAIN_TABLE_JOB, id,
                                     job.project_id, request, self._driver)

    async def validate_custom_train_table(
            self,
            id: GenerateTrainTableJobID,
            custom_table: SourceTableType,
            train_table_mod: TrainingTableSpec,
            extensive_validation: bool = Body(False),
            ctx: RequestContext = Depends(get_ctx),
    ) -> api.ValidationResponse:
        """
        Validate a custom training table against the original training table
        generated via Predictive Query.
        Args:
            id: Id of the original training table generation job.
            custom_table: A SourceTableType with connector and table name of
                the custom training table.
            train_table_mod: What the custom table carries beyond the
                generated columns. A specification naming no weight column
                describes the generated table as it stands, and is checked
                for everything except its weights.
        Returns:
            A ValidationResponse object with the validation result.
        """
        job = await self.get(id, ctx)
        if job.job_status_report.status != api.JobStatus.DONE:
            raise HTTPException(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                'Unable to validate custom training table as training '
                f'table gen job is in status: {job.job_status_report.status}')
        with wrap_driver_errors():
            result = await self._ml_job_backend.get_train_table_result(id)
            return await self._driver.validate_custom_train_table(
                custom_table,
                train_table_mod.weight_col,
                result.train_table_path,
                extensive_validation,
                ProjectID(job.project_id) if job.project_id else None,
                train_table_mod.split_col,
            )
