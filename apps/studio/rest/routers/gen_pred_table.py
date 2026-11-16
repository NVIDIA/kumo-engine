# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from kumoapi.jobs import ErrorDetails, GeneratePredictionTableJobResource

from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.batch_prediction_job import BatchPredictionJobAnchorTime
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.utils.project_scope import (
    require_project_id,
    require_requested_project_visible_for_user,
    require_visible_project_resource_for_user,
)
from rest.utils.public import wrap_rpc_error
from rest.utils.routers import nullable_response, wrap_validation_exceptions
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    JobType,
    ProjectID,
)
from statedb.utils.errors import PredictiveQueryDoesNotExistError

log = logging.getLogger(__name__)


@dataclass
class GeneratePredictionTableResponse:
    # Contains pquery ID, training table generation plan,
    # graph snapshot ID
    resource: GeneratePredictionTableJobResource


class GeneratePredictionTableRouter:
    def __init__(
        self,
        auth: Authentication,
        driver: PQueryApiDriver,
        ml_job_backend: MLJobBackend,
    ) -> None:
        self._driver = driver
        self._auth = auth
        self._ml_job_backend = ml_job_backend

        self.router = APIRouter(
            prefix="/v0/genpredtable_jobs",
            tags=["Jobs"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={HTTPStatus.NOT_FOUND: {
                "description": "Not found"
            }},
        )

        # Resource:
        self.router.add_api_route(
            "/{id}",
            self.get,
            response_model=GeneratePredictionTableResponse,
            methods=['GET'],
        )

        # Errors:
        self.router.add_api_route(
            '/{id}/get_errors',
            self.get_errors,
            methods=['GET'],
            response_model=ErrorDetails,
        )

        # Cancel:
        self.router.add_api_route(
            "/{id}/cancel",
            self.cancel,
            methods=['POST'],
            response_model=bool,
        )

        # Anchor time:
        self.router.add_api_route(
            "/{id}/anchor_time",
            self.get_anchor_time,
            response_model=datetime,
            methods=['GET'],
            responses=nullable_response(),
        )

        self.router.add_api_route(
            "/{id}/entities",
            self.get_num_entities,
            methods=["GET"],
        )

    async def _get_visible_job(
        self,
        id: GeneratePredictionTableJobID,
        *,
        project_id: str,
        ctx: RequestContext,
    ) -> GeneratePredictionTableJobResource:
        with wrap_rpc_error(
            {PredictiveQueryDoesNotExistError:
             HTTPStatus.UNPROCESSABLE_ENTITY}):
            require_requested_project_visible_for_user(
                project_id,
                user_authorization=ctx.user_authorization,
                not_found_error_factory=lambda: HTTPException(
                    HTTPStatus.NOT_FOUND,
                    f'Generate prediction table job {id} is not found.'),
            )
            jobs = await self._ml_job_backend.get_generate_pred_table_jobs(
                {id},
                project_ids={ProjectID(project_id)},
            )
            if not jobs:
                raise HTTPException(
                    HTTPStatus.NOT_FOUND,
                    f'Generate prediction table job {id} is not found.')
            job = require_visible_project_resource_for_user(
                jobs[0],
                requested_project_id=project_id,
                user_authorization=ctx.user_authorization,
                not_found_error_factory=lambda: HTTPException(
                    HTTPStatus.NOT_FOUND,
                    f'Generate prediction table job {id} is not found.'),
            )
            tags = await self._ml_job_backend.get_job_tags(
                JobType.GENERATE_PREDICTION_TABLE_JOB, id)
            job.tags = tags.tags
            return job

    async def get(
        self,
        id: GeneratePredictionTableJobID,
        project_id: str,
        ctx: RequestContext = Depends(get_ctx),
    ) -> GeneratePredictionTableResponse:
        job = await self._get_visible_job(id, project_id=project_id, ctx=ctx)
        pred_table_response = GeneratePredictionTableResponse(resource=job)

        if not pred_table_response.resource.job_status_report.validation_response:
            with wrap_rpc_error({
                    PredictiveQueryDoesNotExistError:
                    HTTPStatus.UNPROCESSABLE_ENTITY
            }):
                val_response = await self._ml_job_backend.get_job_validation_response(
                    id)
                pred_table_response.resource.job_status_report.validation_response = val_response
        return pred_table_response

    async def get_errors(
            self,
            id: GeneratePredictionTableJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> ErrorDetails:
        await self._get_visible_job(id, project_id=project_id, ctx=ctx)
        return await self._driver.get_errors_for_job(id)

    async def cancel(
            self,
            id: GeneratePredictionTableJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> bool:
        project_id = require_project_id(project_id)
        await self._get_visible_job(id, project_id=project_id, ctx=ctx)
        return await self._ml_job_backend.cancel_generate_pred_table_job(id)

    async def get_anchor_time(
            self,
            id: GeneratePredictionTableJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> Optional[datetime]:
        await self._get_visible_job(id, project_id=project_id, ctx=ctx)
        # NOTE this conversion is OK since the ID of a batch prediction job
        # that has prediction table generation run in standalone mode is the
        # the same as that job's ID:
        bp_id = BatchPredictionJobID(id)

        # First try to load from the job directory
        home_dir = await self._ml_job_backend.get_job_home_dir(bp_id)
        anchor_time = await (
            self._driver.get_prediction_anchor_time_from_per_job_dir(
                bp_id,
                home_dir,
            ))
        if anchor_time is not None:
            return anchor_time.anchor_time

        # If not found in job directory, try loading from cache
        anchor_time = await (
            BatchPredictionJobAnchorTime().load_from_cache.coroutine(bp_id))
        return anchor_time.anchor_time

    async def get_num_entities(
            self,
            id: GeneratePredictionTableJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ):
        await self._get_visible_job(id, project_id=project_id, ctx=ctx)
        with wrap_rpc_error(), wrap_validation_exceptions():
            if not GeneratePredictionTableJobID.is_valid_id(id):
                raise ValueError(
                    f"Gen pred table ID {id} is invalid."
                    "A valid ID is prefixed by 'gen-predtable-job-' followed by a hex key. "
                )
        bp_id = BatchPredictionJobID(id)
        home_dir = await self._ml_job_backend.get_job_home_dir(bp_id)
        return await (self._driver.get_prediction_num_entities(
            bp_id,
            home_dir,
        ))
