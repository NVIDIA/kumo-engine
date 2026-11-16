# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends

from rest.backends.ml_backend.data_util import get_job_base_dir, get_job_dir
from rest.config.kumo_tensorboard import (
    GetManyScalarsRequest,
    JobScalarTags,
    ScalarsResponse,
)
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.tensorboard_api_driver import TensorboardApiDriver
from statedb.config.common import TrainingJobID


class TensorboardRouter:
    def __init__(
        self,
        auth: Authentication,
        driver: TensorboardApiDriver,
    ) -> None:
        self._driver = driver
        self._auth = auth

        self.router = APIRouter(
            prefix="/v0/tensorboard",
            tags=["Tensorboard"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={HTTPStatus.NOT_FOUND: {
                "description": "Not found"
            }},
        )

        self.router.add_api_route(
            "/{job_id}/trials",
            self.list_trials,
            response_model=List[str],
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{job_id}/plugin/scalars/tags",
            self.list_scalars_tags,
            response_model=JobScalarTags,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/{job_id}/plugin/scalars/scalars",
            self.get_scalars,
            response_model=ScalarsResponse,
            methods=['GET'],
        )

        self.router.add_api_route(
            "/plugin/scalars/many_scalars",
            self.get_many_scalars,
            response_model=Dict[str, ScalarsResponse],
            methods=['POST'],
        )

        self.router.add_api_route(
            "/{job_id}/plugin/hparams",
            self.get_hparams,
            response_model=str,
            methods=['GET'],
        )

        # TODO(dong) Add driver for scalars multirun.
        # self.router.add_api_route(
        #     "/{id}/plugin/scalars/scalars_multirun",
        #     self.get_scalars_multirun,
        #     response_model=TBD,
        #     methods=['POST'],
        # )

    async def list_trials(
        self, job_id: str,
        ctx: RequestContext = Depends(get_ctx)) -> List[str]:
        r"""Return a list of trials that contains a tensorboard.
        """
        job_id = TrainingJobID(job_id)
        training_job_path = get_job_dir(get_job_base_dir(), job_id)
        return await self._driver.list_trials(training_job_path)

    async def list_scalars_tags(
            self,
            job_id: str,
            trial: Optional[str] = None,
            ctx: RequestContext = Depends(get_ctx),
    ) -> JobScalarTags:
        r"""Return a list of scalar tags for the trial.
        """
        job_id = TrainingJobID(job_id)
        training_job_path = get_job_dir(get_job_base_dir(), job_id)
        return await self._driver.list_scalars_tags(training_job_path, trial)

    async def get_scalars(
        self, job_id: str, trial: str, tag: str,
        ctx: RequestContext = Depends(get_ctx)) -> ScalarsResponse:
        r"""Return a list of scalars for the trail and tag."""
        job_id = TrainingJobID(job_id)
        training_job_path = get_job_dir(get_job_base_dir(), job_id)
        return await self._driver.get_scalars(training_job_path, trial, tag)

    async def get_many_scalars(
        self,
        request: GetManyScalarsRequest,
    ) -> Dict[str, ScalarsResponse]:
        r"""For a given trial, returns a dictionary of tag to list of scalars
        for that tag."""
        training_job_path = get_job_dir(get_job_base_dir(), request.job_id)
        return await self._driver.get_many_scalars(
            training_job_path,
            request.trial,
            request.tags,
        )

    async def get_hparams(
        self, job_id: str, ctx: RequestContext = Depends(get_ctx)) -> str:
        r"""Return the model config for the trail in the json string."""
        job_id = TrainingJobID(job_id)
        training_job_path = get_job_dir(get_job_base_dir(), job_id)
        return await self._driver.get_hparams(training_job_path)
