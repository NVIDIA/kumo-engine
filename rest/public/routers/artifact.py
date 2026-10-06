# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Callable

import kumoapi.jobs as api
from fastapi import APIRouter, Depends, Query
from kumoapi.common import JobStatus
from typing_extensions import Annotated

from io_utils import logging
from rest.config import desc
from rest.config.artifact_export import HostedArtifactExportRequest
from rest.deps.authentication import Authentication
from rest.driver.artifact_api_driver import ArtifactApiDriver
from rest.utils.public import wrap_driver_errors
from statedb.config.common import ProjectID

log = logging.getLogger(__name__)


class ArtifactRouterV1:
    def __init__(
        self,
        api_key_auth: Callable,
        driver: ArtifactApiDriver,
        auth: Authentication,
    ) -> None:
        self._driver = driver
        self._auth = auth

        self.router = APIRouter(
            prefix='/artifact',
            tags=["API v1 (Under Development): Kumo Artifact"],
            dependencies=[Depends(api_key_auth)],
        )

        self.router.add_api_route(
            "",
            self.create_artifact_export_job,
            response_model=api.ArtifactExportResponse,
            summary='Launch an artifact export job.',
            methods=['POST'],
            description=desc.start_pred,
            response_description='Export job summary for the artifact.',
            responses={
                HTTPStatus.UNPROCESSABLE_ENTITY: {
                    'description': "The artifact is missing."
                }
            },
        )

        self.router.add_api_route(
            "/{job_id}",
            self.get_artifact_export_job_status,
            response_model=JobStatus,
            methods=['GET'],
            summary='Get an artifact export job status.',
            response_description='The artifact export job status.',
        )

        self.router.add_api_route(
            "/{job_id}/cancel",
            self.cancel_artifact_export_job,
            response_model=JobStatus,
            methods=['POST'],
            summary='Cancel an artifact export job.',
            response_description=(
                'Cancel and returns the artifact export job status.'),
        )

    async def create_artifact_export_job(
        self,
        request: HostedArtifactExportRequest,
        project_id: Annotated[str, Query()],
    ) -> api.ArtifactExportResponse:
        output_config = request.prediction_output or request.training_table_output
        assert output_config is not None
        with wrap_driver_errors():
            id = await self._driver.create_artifact_export_job(
                request.job_id or '',
                output_config,
                ProjectID(project_id),
                request.request_id,
            )
        return api.ArtifactExportResponse(id)

    async def get_artifact_export_job_status(
        self,
        job_id: str,
        project_id: Annotated[str, Query()],
    ) -> JobStatus:
        with wrap_driver_errors():
            return await self._driver.get_artifact_export_job_status(
                job_id, ProjectID(project_id))

    async def cancel_artifact_export_job(
        self,
        job_id: str,
        project_id: Annotated[str, Query()],
    ) -> JobStatus:
        with wrap_driver_errors():
            return await self._driver.cancel_artifact_export_job(
                job_id, ProjectID(project_id))
