# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from http import HTTPStatus
from typing import Dict, List, Optional, Set, Union

from fastapi import APIRouter, Depends

from io_utils import kumo_dataclass
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config import desc
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.utils.project_scope import require_project_id
from rest.utils.public import wrap_rpc_error
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    JobID,
    JobStatus,
    JobType,
    ProjectID,
    TrainingJobID,
)
from statedb.config.job_tag import (
    JobTagKey,
    JobTagRequest,
    JobTags,
    JobTagValue,
    check_custom_tag_key,
)
from statedb.db.ml_jobs import GenericJob, ListJobsQuery

log = logging.getLogger(__name__)


@kumo_dataclass
class ListJobsResponse:
    jobs: List[GenericJob]
    total_num_items: int


class JobsResource:
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
            prefix="/v0/jobs",
            tags=["Jobs"],
            dependencies=[Depends(auth.authenticate_request_header)],
            responses={HTTPStatus.NOT_FOUND: {
                "description": "Not found"
            }},
        )

        self.router.add_api_route(
            "/",
            self.list,
            response_model=ListJobsResponse,
            methods=['POST'],
            summary='List all jobs running in this environment',
            description=desc.list_all_jobs,
            response_description='List of job objects.',
        )

        self.router.add_api_route(
            "/prediction_tags",
            self.get_all_prediction_tags,
            response_model=Dict[JobID, JobTags],
            methods=['GET'],
            summary='Map all tags for all prediction jobs.',
        )

        self.router.add_api_route(
            "/training_tags",
            self.get_all_training_tags,
            response_model=Dict[JobID, JobTags],
            methods=['GET'],
            summary='Map of all tags for all training jobs.',
        )

        self.router.add_api_route(
            "/get_active_jobs",
            self.get_currently_running_jobs_or_job_status,
            response_model=Dict[Union[TrainingJobID, BatchPredictionJobID,
                                      GenerateTrainTableJobID,
                                      GeneratePredictionTableJobID],
                                Union[JobStatus, GenericJob]],
            methods=['POST'],
            summary='Get jobs that are currently running or queued.',
        )

        self.router.add_api_route(
            "/{job_id}/archive",
            self.archive_job,
            response_model=bool,
            methods=['POST'],
            summary='Archive a job.',
        )

        self.router.add_api_route(
            "/add_tag",
            self.add_job_tag,
            response_model=bool,
            methods=['POST'],
            summary='Add a tag associated with a job.',
        )

        self.router.add_api_route(
            "/delete_tag",
            self.delete_job_tag,
            response_model=bool,
            methods=['POST'],
            summary='Remove a tag associated with a job.',
        )

    @staticmethod
    def _filter_visible_jobs(
        jobs: List[GenericJob],
        *,
        ctx: RequestContext,
        requested_project_id: ProjectID,
    ) -> List[GenericJob]:
        user_authorization = ctx.user_authorization
        if user_authorization.is_admin:
            return jobs
        if requested_project_id not in user_authorization.project_ids:
            return []
        visible_project_ids = set(user_authorization.project_ids)
        return [
            job for job in jobs
            if getattr(job, "project_id", None) in visible_project_ids
        ]

    @staticmethod
    def _requested_project_visible(
        *,
        ctx: RequestContext,
        requested_project_id: ProjectID,
    ) -> bool:
        user_authorization = ctx.user_authorization
        return (user_authorization.is_admin
                or requested_project_id in user_authorization.project_ids)

    async def list(
            self,
            list_query: ListJobsQuery,
            ctx: RequestContext = Depends(get_ctx),
    ) -> ListJobsResponse:
        """
        List all SDK jobs running in Kumo.

        Limitations: Graph refresh or snapshots are not listed here yet.

        Args:
            list_query (ListJobsQuery): Limits and filters
            ctx (RequestContext, optional): The auth context.

        Returns:
            List[GenericJob]: List of all jobs
        """
        if not self._requested_project_visible(
                ctx=ctx, requested_project_id=list_query.project_id):
            return ListJobsResponse(jobs=[], total_num_items=0)

        # Get all job tags
        job_tags: Dict[JobID, JobTags] = {}
        for job_type in set(list_query.job_type):
            job_tags_by_type: Dict[JobID, JobTags] = (
                await
                self._ml_job_backend.list_job_tags(job_type, tag_filters={}))
            job_tags.update(job_tags_by_type)

        # Get all jobs from ML Backend and apply filter inside the ML Backend
        # to speed up this endpoint by preventing unnecessary serialization
        # and deserialization of jobs across the temporal RPC boundary
        with wrap_rpc_error():
            jobs, num_items = (await self._ml_job_backend.get_all_ml_jobs(
                ctx, list_query, job_tags))
        jobs = self._filter_visible_jobs(
            jobs,
            ctx=ctx,
            requested_project_id=list_query.project_id,
        )

        return ListJobsResponse(
            jobs=jobs,
            total_num_items=num_items,
        )

    async def get_all_prediction_tags(
            self,
            project_id: ProjectID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> Dict[JobID, JobTags]:
        # Get job tags for all prediction and prediction table jobs
        if not self._requested_project_visible(ctx=ctx,
                                               requested_project_id=project_id):
            return {}
        job_tags: Dict[JobID, JobTags] = {}
        for job_type in (JobType.BATCH_PREDICTION_JOB,
                         JobType.GENERATE_PREDICTION_TABLE_JOB):
            job_tags_by_type: Dict[JobID, JobTags] = (
                await self._ml_job_backend.list_project_job_tags(
                    job_type, project_id, tag_filters={}))
            job_tags.update(job_tags_by_type)
        return job_tags

    async def get_all_training_tags(
            self,
            project_id: ProjectID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> Dict[JobID, JobTags]:
        # Get job tags for all training and training table jobs
        if not self._requested_project_visible(ctx=ctx,
                                               requested_project_id=project_id):
            return {}
        job_tags: Dict[JobID, JobTags] = {}
        for job_type in (JobType.GENERATE_TRAIN_TABLE_JOB,
                         JobType.TRAINING_JOB):
            job_tags_by_type: Dict[JobID, JobTags] = (
                await self._ml_job_backend.list_project_job_tags(
                    job_type, project_id, tag_filters={}))
            job_tags.update(job_tags_by_type)
        return job_tags

    async def get_currently_running_jobs_or_job_status(
        self,
        project_id: ProjectID,
        job_ids: Optional[Set[JobID]] = None,
        ctx: RequestContext = Depends(get_ctx),
    ) -> Dict[Union[TrainingJobID, BatchPredictionJobID,
                    GenerateTrainTableJobID, GeneratePredictionTableJobID],
              Union[JobStatus, GenericJob]]:
        """
        Endpoint to fetch only jobs or job status for jobs
        that are currently running or queued.
        Args:
            job_ids: An optional array of job ids for
            jobs that are running or queued and their metadata
            is already loaded in the UI. Only return the
            job status for these IDs.
            ctx (RequestContext): The auth context.

        Returns:
            Dict[JobID, GenericJob | JobStatus]: If no jobs are currently
            running or are queued, return an empty dictionary.
            Otherwise, return a mapping of all the running jobs.
        """
        if not self._requested_project_visible(ctx=ctx,
                                               requested_project_id=project_id):
            return {}
        with wrap_rpc_error():
            return await self._ml_job_backend.get_in_progress_job_or_job_status(
                ctx, job_ids, project_id=project_id)

    async def archive_job(
            self,
            job_id: JobID,
            archived: bool,
            project_id: ProjectID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> bool:
        """
        Archive a job.

        Args:
            job_id: The ID of the job to archive (from URL path)
            archived: Boolean flag indicating whether the job should be archived
            ctx (RequestContext): The auth context.

        Returns:
            bool: True if job was archived, False otherwise
        """
        project_id = ProjectID(require_project_id(project_id))
        if not self._requested_project_visible(ctx=ctx,
                                               requested_project_id=project_id):
            return False
        with wrap_rpc_error():
            return await self._ml_job_backend.archive_job(
                job_id,
                archived,
                project_id=project_id,
            )

    async def add_job_tag(
            self,
            request: JobTagRequest,
            ctx: RequestContext = Depends(get_ctx),
    ) -> bool:
        """
        Add a tag to job.

        Args:
            job_id: The ID of the job the tag is associated with.
            tag: The key value of the tag.
        Returns:
            bool: True if the tag was successfully added and False if it wasn't.
        """
        if not self._requested_project_visible(
                ctx=ctx, requested_project_id=request.project_id):
            return False
        job_type = JobType.from_id(request.job_id)
        key = JobTagKey(request.key)
        try:
            check_custom_tag_key(key)
        except ValueError:
            return False
        key_dict = {
            key:
            JobTagValue(request.value)
            if request.value else JobTagValue(request.key)
        }
        return await self._ml_job_backend.update_job_tags(
            job_type,
            request.job_id,
            key_dict,
            project_id=request.project_id,
        )

    async def delete_job_tag(
            self,
            request: JobTagRequest,
            ctx: RequestContext = Depends(get_ctx),
    ) -> bool:
        """
        Delete a tag from job.

        Args:
            job_id: The ID of the job the tag is associated with.
            tag: The key value of the tag.
        Returns:
            bool: True if the tag was successfully deleted and False if it wasn't.
        """
        if not self._requested_project_visible(
                ctx=ctx, requested_project_id=request.project_id):
            return False
        job_type = JobType.from_id(request.job_id)
        return await self._ml_job_backend.delete_job_tag(
            job_type,
            request.job_id,
            JobTagKey(request.key),
            project_id=request.project_id,
        )
