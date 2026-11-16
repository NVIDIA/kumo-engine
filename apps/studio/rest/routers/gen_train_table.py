# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from kumoapi.jobs import ErrorDetails, GenerateTrainTableJobResource

from kumo.stats.train_table_stats import (
    SplitTrainTableStats,
    TrainTableGenerationProgress,
)
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.utils.project_scope import (
    require_project_id,
    require_requested_project_visible_for_user,
    require_visible_project_resource_for_user,
)
from rest.utils.public import wrap_rpc_error
from rest.utils.routers import (
    ValidationHTTPException,
    nullable_response,
    wrap_validation_exceptions,
)
from statedb.config.common import GenerateTrainTableJobID, JobType, ProjectID
from statedb.utils.errors import PredictiveQueryDoesNotExistError

log = logging.getLogger(__name__)


@dataclass
class GenerateTrainTableResponse:
    # Contains pquery ID, training table generation plan,
    # graph snapshot ID
    resource: GenerateTrainTableJobResource


class GenerateTrainTableRouter:
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
            prefix="/v0/gentraintable_jobs",
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
            response_model=GenerateTrainTableResponse,
            methods=['GET'],
        )

        # Progress of training table generation:
        self.router.add_api_route(
            "/{id}/generation_stats",
            self.get_generation_stats,
            response_model=TrainTableGenerationProgress,
            methods=['GET'],
            responses=nullable_response(),
        )

        # Split statistics:
        self.router.add_api_route(
            "/{id}/split_stats",
            self.get_split_stats,
            # Optional: LP (and other pre-stats states) legitimately have no
            # per-stage split stats -- the handler returns None there, which a
            # non-Optional response_model would 500 on despite nullable_response
            # only affecting the schema docs, not runtime validation.
            response_model=Optional[SplitTrainTableStats],
            methods=['GET'],
            responses=nullable_response(),
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

    async def _get_visible_job(
        self,
        id: GenerateTrainTableJobID,
        *,
        project_id: Optional[str],
        ctx: RequestContext,
    ) -> GenerateTrainTableJobResource:
        with wrap_rpc_error(
            {PredictiveQueryDoesNotExistError:
             HTTPStatus.UNPROCESSABLE_ENTITY}):
            require_requested_project_visible_for_user(
                project_id,
                user_authorization=ctx.user_authorization,
                not_found_error_factory=lambda: HTTPException(
                    HTTPStatus.NOT_FOUND,
                    f'Generate train table job {id} is not found.'),
            )
            jobs = await self._ml_job_backend.get_generate_train_table_jobs(
                {id},
                project_ids={ProjectID(project_id)},
            )
            if not jobs:
                raise HTTPException(
                    HTTPStatus.NOT_FOUND,
                    f'Generate train table job {id} is not found.')
            job = require_visible_project_resource_for_user(
                jobs[0],
                requested_project_id=project_id,
                user_authorization=ctx.user_authorization,
                not_found_error_factory=lambda: HTTPException(
                    HTTPStatus.NOT_FOUND,
                    f'Generate train table job {id} is not found.'),
            )
            tags = await self._ml_job_backend.get_job_tags(
                JobType.GENERATE_TRAIN_TABLE_JOB, id)
            job.tags = tags.tags
            return job

    async def get(
            self,
            id: GenerateTrainTableJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> GenerateTrainTableResponse:
        job = await self._get_visible_job(id, project_id=project_id, ctx=ctx)
        return GenerateTrainTableResponse(resource=job)

    async def get_errors(
            self,
            id: GenerateTrainTableJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> ErrorDetails:
        await self._get_visible_job(id, project_id=project_id, ctx=ctx)
        return await self._driver.get_errors_for_job(id)

    async def get_generation_stats(
        self, id: GenerateTrainTableJobID, project_id: str,
        ctx: RequestContext = Depends(get_ctx)
    ) -> Optional[TrainTableGenerationProgress]:
        r"""Fetch training table generation statistics.

        Args:
            id: the ID of the training table generation job.

        Raises:
            ValueError: if the query does not exist or is not yet trained.
        """
        try:
            with wrap_validation_exceptions(PredictiveQueryDoesNotExistError):
                await self._get_visible_job(id, project_id=project_id, ctx=ctx)
                return await self._driver.get_train_table_generation_stats(id)
        except (ValueError, KeyError, NotImplementedError) as err:
            raise ValidationHTTPException(str(err), err)

    async def get_split_stats(
        self,
        id: GenerateTrainTableJobID,
        project_id: str,
        ctx: RequestContext = Depends(get_ctx),
    ) -> Optional[SplitTrainTableStats]:
        r"""Get split train table statistics.

        Args:
            id: the ID of the training table generation job

        Returns:
            SplitTrainTableStats class which has split train table statistics.

        Raises:
            - ValueError: if the query does not exists, or not yet trained
        """
        try:
            with wrap_validation_exceptions(PredictiveQueryDoesNotExistError):
                await self._get_visible_job(id, project_id=project_id, ctx=ctx)
                response = await self._driver.get_train_table_split_stats(id)
                return response
        except (ValueError, KeyError, NotImplementedError) as err:
            raise ValidationHTTPException(str(err), err)

    async def cancel(
            self,
            id: GenerateTrainTableJobID,
            project_id: str,
            ctx: RequestContext = Depends(get_ctx),
    ) -> bool:
        project_id = require_project_id(project_id)
        await self._get_visible_job(id, project_id=project_id, ctx=ctx)
        return await self._ml_job_backend.cancel_generate_train_table_job(id)
