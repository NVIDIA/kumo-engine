# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from http import HTTPStatus
from typing import Annotated, Any, Callable, Dict, List, Optional

import kumoapi.jobs as api
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse
from kumo_core.model_plan import ModelPlan, ModelPlanValueError
from kumo_core.model_plan_conversion import (
    internal_model_plan_from_public,
    public_model_plan_from_internal,
)

from io_utils import fsutil, logging
from kumo.home import HomeDirManager
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config import desc
from rest.config.status import EmptyResponse, IDResponse, PatchResponse
from rest.config.training_job import TrainingJobResponse
from rest.config.training_job_v2 import TrainingJobInternalRequest
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.public.routers.common import (
    export_training_metrics,
    filter_by_project,
    list_job_tags,
    public_job_failure_validation,
    require_project_id_valid,
    resolve_requested_project,
    update_job_tags,
    validate_custom_job_tags,
)
from rest.utils.public import (
    get_base_kumo_url,
    wrap_driver_errors,
    wrap_rpc_error,
)
from rest.utils.routers import nullable_response
from statedb.config.common import (
    JobSource,
    JobStatus,
    JobType,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.config.job_tag import TAG_KEY_PQUERY_NAME, JobTags
from statedb.config.timestamp import get_current_timestamps
from statedb.utils.errors import PredictiveQueryDoesNotExistError

log = logging.getLogger(__name__)


def _submission_time(job_response: Any) -> datetime:
    """Return submission time for StateDB entries and backend responses."""
    timestamps = getattr(job_response, "timestamps", None)
    if timestamps is not None:
        return timestamps.created_at

    created_at = getattr(job_response, "created_at", None)
    if created_at is not None:
        return created_at

    # Compatibility for hand-built legacy response objects without either
    # submission-time representation.
    return (job_response.execution.start_time
            or job_response.execution.last_updated_at)


class TrainingRouterV1:
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
            prefix='/training_jobs',
            tags=["API v1 (Under Development): Training Jobs"],
            dependencies=[Depends(api_key_auth)],
        )
        self.router.add_api_route(
            '',
            self.create,
            response_model=IDResponse,
            methods=['POST'],
            summary='Launch a training job.',
            description=desc.start_training,
            responses={
                HTTPStatus.BAD_REQUEST: {
                    'description':
                    "Cannot start a new training while one is still running."
                }
            },
        )

        self.router.add_api_route(
            "",
            self.list,
            response_model=List[api.TrainingJobResource],
            methods=['GET'],
            summary='List training jobs of a Predictive Query.',
            description=desc.list_train,
            response_description='List of TrainingJobResource objects.',
        )
        self.router.add_api_route(
            "/{training_job_id}",
            self.get,
            response_model=api.TrainingJobResource,
            methods=['GET'],
            summary='Get the training job status of a Predictive Query.',
            description=desc.get_train,
            response_description='TrainingJobResource object that summarizes '
            'the training status of a Predictive Query, and the training '
            'result summary if training has completed successfully.',
        )
        self.router.add_api_route(
            '/{training_job_id}/holdout',
            self.get_holdout_df,
            response_class=PlainTextResponse,
            methods=['GET'],
            summary=(
                'Get holdout dataset URL for a successfully completed model '
                'training job.'),
        )
        self.router.add_api_route(
            "/{training_job_id}/cancel",
            self.cancel,
            response_model=api.CancelTrainingJobResponse,
            methods=['POST'],
            summary='Cancel a query (re)training job.',
            description=desc.cancel_train,
            response_description='Empty',
        )
        self.router.add_api_route(
            "/{training_job_id}",
            self.delete,
            response_model=EmptyResponse,
            methods=['DELETE'],
            summary='Delete a finished query (re)training job including job '
            'metadata as well as model artifacts if exist.',
            description=desc.delete_train,
            response_description='Empty',
        )

        self.router.add_api_route(
            '/{training_job_id}/tags',
            self.update_tags,
            methods=['PATCH'],
            response_model=PatchResponse,
            summary='Update(add/mutate/remove) job tags.',
        )

        self.router.add_api_route(
            '/{training_job_id}/progress',
            self.get_progress,
            methods=['GET'],
            response_model=api.AutoTrainerProgress,
        )

        self.router.add_api_route(
            "/{training_job_id}/encoders",
            self.get_encoders_for_training_job,
            response_model=Optional[Dict[str, str]],
            methods=['GET'],
            responses={
                **nullable_response(),
                HTTPStatus.NOT_FOUND: {
                    'description':
                    "The training job does not exist, or it failed or was "
                    "cancelled without producing encoders."
                },
            },
        )

    async def get_progress(
            self,
            training_job_id: TrainingJobID,
            project_id: Annotated[str, Query()],
            ctx: RequestContext = Depends(get_ctx),
    ) -> api.AutoTrainerProgress:
        with wrap_driver_errors():
            return await self._driver.get_training_progress(
                ctx, training_job_id, ProjectID(project_id))

    async def create(
            self,
            request: api.TrainingJobRequest,
            project_id: Annotated[str, Query()],
            ctx: RequestContext = Depends(get_ctx),
    ) -> IDResponse:
        validate_custom_job_tags(request.job_tags)
        require_project_id_valid(ctx, project_id)
        reserved_training_job_id = TrainingJobID.make_random_id()
        with wrap_driver_errors({
                PredictiveQueryDoesNotExistError:
                HTTPStatus.UNPROCESSABLE_ENTITY,
                ModelPlanValueError:
                HTTPStatus.UNPROCESSABLE_ENTITY,
        }):
            job_id = await self._driver.start_training_job_v2(
                ctx=ctx,
                request=TrainingJobInternalRequest(
                    pquery_id=request.pquery_id,
                    model_plan=internal_model_plan_from_public(
                        request.model_plan),
                    job_tags=request.job_tags,
                    graph_snapshot_id=request.graph_snapshot_id,
                    train_table_job_id=request.train_table_job_id,
                    custom_train_table=request.custom_train_table,
                    warm_start_job_id=request.warm_start_job_id,
                ),
                job_source=JobSource.SDK,
                reserved_training_job_id=reserved_training_job_id,
                project_id=ProjectID(project_id),
            )

        return IDResponse(job_id)

    async def get(
            self,
            training_job_id: TrainingJobID,
            project_id: Annotated[Optional[str], Query()] = None,
            ctx: RequestContext = Depends(get_ctx),
    ) -> api.TrainingJobResource:
        with wrap_driver_errors():
            training_job = await self._driver.get_training_job(
                ctx,
                training_job_id,
                strip_expensive_fields=False,
                project_id=(ProjectID(project_id)
                            if project_id is not None else None),
            )

        if training_job is None:
            raise HTTPException(
                HTTPStatus.NOT_FOUND,
                f'Training job {training_job_id} is not found.')

        tags = JobTags(
            timestamps=get_current_timestamps(),
            tags=training_job.tags or {},
        )
        return self._make_training_job_resource_v1(training_job, tags)

    async def get_holdout_df(
            self,
            training_job_id: TrainingJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> str:
        training_job = await self.get(training_job_id, ctx=ctx)

        if training_job.job_status_report.status != api.JobStatus.DONE:
            raise HTTPException(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                f'Unable to access holdout data for training job in status: '
                f'{training_job.job_status_report.status}')

        with wrap_rpc_error():
            job_home_dir = await self._ml_job_backend.get_job_home_dir(
                training_job_id)

        home_dir_mgr = HomeDirManager(job_home_dir)
        task_name = training_job.tags.get(TAG_KEY_PQUERY_NAME) or 'NA'
        best_trainer_dir = await home_dir_mgr.get_best_trainer_dir.coroutine(
            task_name)
        holdout_df_path = os.path.join(best_trainer_dir, 'test_df.parquet')
        if not await fsutil.exists_async(holdout_df_path):
            log.error('Holdout df does not exist: %s', holdout_df_path)
            raise HTTPException(
                HTTPStatus.NOT_FOUND,
                f'Holdout dataset is missing for model training job, '
                f'id={training_job_id}')

        helper = fsutil.get_presigned_helper(holdout_df_path)
        if helper is not None:
            return await helper.generate_download_url.coroutine(holdout_df_path)
        return fsutil.customer_facing_path(holdout_df_path)

    async def cancel(
        self,
        training_job_id: TrainingJobID,
        project_id: Annotated[str, Query()],
        ctx: RequestContext = Depends(get_ctx),
    ) -> api.CancelTrainingJobResponse:
        log.info('Send cancellation request for job: %s', training_job_id)
        with wrap_driver_errors():
            is_cancelled = await self._driver.cancel_training_job(
                ctx, training_job_id, ProjectID(project_id))
            return api.CancelTrainingJobResponse(is_cancelled)

    async def delete(
            self,
            training_job_id: TrainingJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> EmptyResponse:
        log.info('Deleting job: %s', training_job_id)
        job = await self.get(training_job_id, ctx=ctx)
        with wrap_driver_errors():
            await self._driver.delete_training_job(
                ctx,
                PredictiveQueryID.parse(job.config.pquery_id)[0],
                training_job_id)
        return EmptyResponse()

    async def list(
        self,
        request: Request,
        response: Response,
        pquery_id: Optional[PredictiveQueryID] = None,
        pquery_name: Optional[PredictiveQueryID] = None,
        job_status: Optional[JobStatus] = None,
        limit: Annotated[Optional[int], Query(ge=1)] = None,
        project_id: Annotated[Optional[str], Query()] = None,
        ctx: RequestContext = Depends(get_ctx),
    ) -> List[api.TrainingJobResource]:
        requested_project, project_scope_matches = await resolve_requested_project(
            ctx, self._driver, project_id)
        job_tags = await list_job_tags(JobType.TRAINING_JOB, request,
                                       self._driver, response)
        if not project_scope_matches:
            return []

        job_ids = set(TrainingJobID(id) for id in job_tags)

        with wrap_rpc_error():
            raw_jobs = await self._ml_job_backend.get_training_jobs(
                ctx,
                job_ids,
                job_status,
                limit,
                project_id=requested_project,
            )

        filtered_jobs = filter_by_project(list(raw_jobs.values()), ctx,
                                          lambda j: j.project_id)
        jobs = [
            self._make_training_job_resource_v1(job, job_tags[job.id])
            for job in filtered_jobs
        ]
        jobs.reverse()  # We need reverse chronological order.
        return jobs

    def _make_training_job_resource_v1(
            self, job_response: TrainingJobResponse,
            tags: JobTags) -> api.TrainingJobResource:
        # Convert internal TrainingJobOutput to public TrainingJobResource
        created_at = _submission_time(job_response)
        job_response.pquery_id = job_response.pquery_id or tags.pquery_name
        if not job_response.pquery_def_fpid and tags.pquery_id:
            job_response.pquery_def_fpid = tags.pquery_id.fpid

        job_summary = None
        job = job_response.execution
        if job.status == JobStatus.DONE:
            assert job.results
            job_summary = api.TrainingJobSummary(
                total_elapsed_time=job.last_updated_at - job.start_time,
                automl_experiments_completed=job.results.total_iterations,
                eval_metrics=export_training_metrics(job.results))

        tracking_url = (f'{get_base_kumo_url()}/'
                        f'jobs/training/{job_response.id}')

        model_plan = ModelPlan()
        if isinstance(job_response.advanced_options, ModelPlan):
            model_plan = job_response.advanced_options

        public_model_plan = public_model_plan_from_internal(model_plan)

        # NOTE the TrainingJobResponse dataclass has two fields related to the
        # predictive query ID: 'pquery_id' which should only be set by jobs
        # created in the UI with a pre-defined predictive query ID, and
        # 'pquery_def_fpid' which should be set by jobs created in the SDK
        # (for now; UI later).
        #
        # Here, we set the pquery ID to either be the pquery name (UI) or the
        # public version of the pquery FPID (SDK):
        if job_response.pquery_def_fpid is not None:
            pq_id = PredictiveQueryID.public(job_response.pquery_def_fpid)
        else:
            if job_response.pquery_id is None:
                raise ValueError(
                    f"Job {job_response} does not have a query name "
                    f"or frozen predictive query (definition) fingerprint ID.")
            pq_id = job_response.pquery_id

        return api.TrainingJobResource(
            job_id=job_response.id,
            project_id=job_response.project_id,
            config=api.TrainingJobRequest(
                {},
                pquery_id=pq_id,
                model_plan=public_model_plan,
                # TODO: Populate "graph_snapshot_id" field
                train_table_job_id=job_response.generate_train_table_job_id,
                warm_start_job_id=job_response.warm_start_job_id,
            ),
            job_status_report=api.JobStatusReport(
                status=job.status, tracking_url=tracking_url,
                # The public API schema currently requires this timestamp.
                # Preserve its legacy submission-time value until kumo-api
                # supports an optional execution-start field.
                start_time=(job.start_time
                            or created_at
                            or job.last_updated_at),
                end_time=(job.last_updated_at if job.status.is_terminal else
                          None), event_log=[job.log],
                validation_response=public_job_failure_validation(
                    job_response.execution.failure_info,
                    title='Training failed',
                )),
            result=job_summary,
            created_at=created_at,
            tags=tags.tags,
        )

    async def update_tags(
            self,
            training_job_id: TrainingJobID,
            request: Request,
            ctx: RequestContext = Depends(get_ctx),
    ) -> PatchResponse:
        job = await self.get(training_job_id, ctx=ctx)
        return await update_job_tags(JobType.TRAINING_JOB, training_job_id,
                                     job.project_id, request, self._driver)

    async def get_encoders_for_training_job(
        self, training_job_id: TrainingJobID,
        ctx: RequestContext = Depends(get_ctx)
    ) -> Optional[Dict[str, str]]:
        """
        Returns the encoders used for a specific training job.

        Output: Dictionary mapping from Fully Qualified Table Name to Encoder
        object representation

        {
            connector_id.table_id.column_name: Encoder(p1=2, p2=[41])
        }

        The encoder is meant to be used in the Model Planner and can
        be parsed by ModelPlan.parse_encoder_overrides
        into an encoder object in code

        Persisted encoders are returned regardless of job status. A missing
        training job, or a failed or cancelled job with no encoder artifact,
        returns 404. Missing artifacts for other states return null. Backend
        retrieval failures are propagated.
        """
        with wrap_driver_errors():
            return await self._driver.get_encoders_for_training_job(
                ctx, training_job_id)
