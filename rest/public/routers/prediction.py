# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from http import HTTPStatus
from typing import Annotated, Callable, Dict, List, Optional, Tuple

import kumoapi.jobs as api
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from kumo_core.types import PredictionArtifactType

from io_utils import logging
from kumo.validation import ValidationResponse
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config import desc
from rest.config.batch_prediction_job import BatchPredictionJobResponse
from rest.config.prediction_output import PredictionOutputConfig
from rest.config.status import EmptyResponse, PatchResponse
from rest.deps.authentication import Authentication, RequestContext, get_ctx
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.public.routers.common import (
    filter_by_project,
    get_tag_filters,
    public_job_failure_validation,
    require_project_id_valid,
    resolve_requested_project,
    update_job_tags,
    validate_custom_job_tags,
)
from rest.utils.public import get_base_kumo_url, wrap_driver_errors
from rest.utils.routers import (
    convert_500_http_error_to_503,
    wrap_validation_exceptions,
)
from statedb.config.common import (
    BatchPredictionJobID,
    DataSnapshot,
    GeneratePredictionTableJobID,
    GraphIDv2,
    JobSource,
    JobStatus,
    JobType,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.utils.errors import PredictiveQueryDoesNotExistError

log = logging.getLogger(__name__)

_PUBLIC_PREDICTION_OUTPUT_TYPES: Dict[api.PredictionStorageType, type] = {
    api.PredictionStorageType.SNOWFLAKE: api.SnowflakePredictionOutput,
    api.PredictionStorageType.S3: api.S3PredictionOutput,
    api.PredictionStorageType.GCS: api.GCSPredictionOutput,
    api.PredictionStorageType.ADLS: api.ADLSPredictionOutput,
    api.PredictionStorageType.DATABRICKS: api.DatabricksPredictionOutput,
    api.PredictionStorageType.BIGQUERY: api.BigQueryPredictionOutput,
    api.PredictionStorageType.LOCAL: api.LocalPredictionOutput,
}


def _public_prediction_output(
        output: PredictionOutputConfig) -> api.PredictionOutputConfig:
    r"""Project a persisted output onto its exact public API shape."""
    storage_type = api.PredictionStorageType(output.storage_type)
    public_output_type = _PUBLIC_PREDICTION_OUTPUT_TYPES.get(storage_type)
    if public_output_type is None:
        raise ValueError(
            f'Unsupported persisted prediction output type: {storage_type}')

    try:
        values = {
            field.name: getattr(output, field.name)
            for field in dataclasses.fields(public_output_type)
        }
    except AttributeError as error:
        raise ValueError(
            f'Persisted {storage_type} output is missing a required public '
            'field') from error
    return public_output_type(**values)


class PredictionRouterV1:
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
            prefix='/prediction_jobs',
            tags=["API v1 (Under Development): Batch Prediction Jobs"],
            dependencies=[Depends(api_key_auth)],
        )
        self.router.add_api_route(
            "",
            self.create_prediction_job,
            methods=['POST'],
            summary='Launch a Batch Prediction job with a Predictive Query.',
            description=desc.start_pred,
            response_description='Batch Prediction job summary for the query.',
            responses={
                HTTPStatus.BAD_REQUEST: {
                    'description': "Predictive Query must be trained first."
                }
            },
        )
        self.router.add_api_route(
            "",
            self.list_prediction_jobs,
            response_model=List[api.BatchPredictionJobResource],
            methods=['GET'],
            summary='List the Batch Prediction jobs.',
            description=desc.list_pred,
            response_description='List of BatchPredictionJobResource objects',
        )
        self.router.add_api_route(
            "/{job_id}",
            self.get_prediction_job,
            response_model=api.BatchPredictionJobResource,
            methods=['GET'],
            summary='Get the Batch Prediction job status of '
            'a Predictive Query.',
            description=desc.pred_status,
            response_description='BatchPredictionJobResource object that '
            'represents the current batch prediction job status of a '
            'predictive query.',
        )
        self.router.add_api_route(
            '/{job_id}/progress',
            self.get_progress,
            methods=['GET'],
            summary='Get the Batch Prediction job progress.',
            description=desc.progress,
            response_description='PredictionProgress object that represents '
            'the current batch prediction job progress of a predictive '
            'query.',
            response_model=api.PredictionProgress,
        )
        self.router.add_api_route(
            "/{job_id}/cancel",
            self.cancel_prediction_job,
            response_model=api.CancelBatchPredictionJobResponse,
            methods=['POST'],
            summary='Cancel a Batch Prediction job.',
            description=desc.cancel_pred,
            response_description='Empty',
        )
        self.router.add_api_route(
            "/{job_id}",
            self.delete_prediction_job,
            response_model=EmptyResponse,
            methods=['DELETE'],
            summary='Delete a finished batch prediction job including job '
            'metadata as well as kumo-stored output result if exist.',
            description=desc.delete_pred,
            response_description='Empty',
        )
        self.router.add_api_route(
            '/{id}/tags',
            self.update_tags,
            methods=['PATCH'],
            response_model=PatchResponse,
            summary='Update(add/mutate/remove) job tags.',
        )
        self.router.add_api_route(
            '/{job_id}/get_prediction_df_urls',
            self.get_prediction_df_urls,
            methods=['GET'],
            response_model=api.GetPredictionsDfUrlResponse,
            summary='Returns a list of presigned URLs to directly access the '
            'predictions',
        )
        self.router.add_api_route(
            '/{job_id}/get_embedding_df_urls',
            self.get_embedding_df_urls,
            methods=['GET'],
            response_model=api.GetEmbeddingsDfUrlResponse,
            summary='Returns a list of presigned URLs to directly access the '
            'embeddings',
        )

    async def get_prediction_df_urls(
        self, job_id: BatchPredictionJobID,
        ctx: RequestContext = Depends(get_ctx)
    ) -> api.GetPredictionsDfUrlResponse:
        """Generates presigned URLs to download the predictions DataFrame.

        Args:
            job_id (BatchPredictionJobID): ID of the BP job for which the
                predictions DataFrame URLs are requested.
        """
        with wrap_driver_errors(), wrap_validation_exceptions():
            response = await self._driver.get_presigned_prediction_urls(
                ctx, job_id)

        if response is None:
            raise HTTPException(
                HTTPStatus.NOT_FOUND,
                f'No predictions found for batch prediction job {job_id}.')

        return response

    async def get_embedding_df_urls(
        self, job_id: BatchPredictionJobID,
        ctx: RequestContext = Depends(get_ctx)
    ) -> api.GetEmbeddingsDfUrlResponse:
        """Generates presigned URLs to download the embeddings DataFrame.

        Args:
            job_id (BatchPredictionJobID): ID of the BP job for which the
                embeddings DataFrame URLs are requested.
        """
        with wrap_driver_errors(), wrap_validation_exceptions():
            response = await self._driver.get_presigned_embedding_urls(
                ctx, job_id)

        if response is None:
            raise HTTPException(
                HTTPStatus.NOT_FOUND,
                f'No embeddings found for batch prediction job {job_id}.')

        return response

    async def create_prediction_job(
        self,
        request: api.BatchPredictionRequest,
        project_id: Annotated[str, Query()],
        ctx: RequestContext = Depends(get_ctx),
    ) -> Tuple[Optional[str], ValidationResponse]:
        # Ensure S3 outputs are treated as directories by appending a trailing
        # slash when missing. This mirrors the v0 behavior.
        request = dataclasses.replace(
            request, outputs=_require_partitioned_s3_output(request.outputs))
        validate_custom_job_tags(request.job_tags)
        require_project_id_valid(ctx, project_id)
        with wrap_driver_errors():
            job_id, validation_response = (
                await self._driver.start_batch_prediction_job_v2(
                    ctx,
                    request,
                    job_source=JobSource.SDK,
                    project_id=ProjectID(project_id),
                ))
        return (job_id, validation_response)

    async def get_prediction_job(
        self,
        job_id: BatchPredictionJobID,
        project_id: Annotated[Optional[str], Query()] = None,
        ctx: RequestContext = Depends(get_ctx),
    ) -> api.BatchPredictionJobResource:
        with convert_500_http_error_to_503(), wrap_driver_errors():
            job = await self._driver.get_batch_prediction_job(
                ctx,
                job_id,
                project_id=(ProjectID(project_id)
                            if project_id is not None else None),
            )

        if job is None:
            raise HTTPException(HTTPStatus.NOT_FOUND,
                                f'No batch prediction job found for: {job_id}')
        return _make_prediction_job_resource(job)

    async def get_progress(
            self,
            job_id: BatchPredictionJobID,
            project_id: Annotated[str, Query()],
            ctx: RequestContext = Depends(get_ctx),
    ) -> api.PredictionProgress:
        with wrap_driver_errors():
            progress = await self._driver.get_batch_prediction_progress(
                ctx, job_id, ProjectID(project_id))

        if progress is None:
            return api.PredictionProgress()
        return progress

    async def list_prediction_jobs(
        self,
        request: Request,
        response: Response,
        model_id: Optional[TrainingJobID] = None,
        pquery_name: Optional[PredictiveQueryID] = None,
        pquery_id: Optional[PredictiveQueryID] = None,
        job_status: Optional[JobStatus] = None,
        limit: Annotated[Optional[int], Query(ge=1)] = None,
        project_id: Annotated[Optional[str], Query()] = None,
        ctx: RequestContext = Depends(get_ctx),
    ) -> List[api.BatchPredictionJobResource]:
        requested_project, project_scope_matches = await resolve_requested_project(
            ctx, self._driver, project_id)
        tag_filters = get_tag_filters(request, response)
        if not project_scope_matches:
            return []
        with wrap_driver_errors(
            {PredictiveQueryDoesNotExistError: HTTPStatus.NOT_FOUND}):
            jobs = await self._driver.list_batch_prediction_jobs(
                ctx,
                job_status=job_status,
                limit=limit,
                tag_filters=tag_filters,
                project_id=requested_project,
            )

        jobs = [
            job for job in jobs
            if job_status is None or job.execution.status == job_status
        ]
        jobs = filter_by_project(jobs, ctx, lambda j: j.project_id)
        jobs.reverse()  # We need reverse chronological order.
        return [_make_prediction_job_resource(job) for job in jobs[:limit]]

    async def cancel_prediction_job(
        self,
        job_id: BatchPredictionJobID,
        project_id: Annotated[str, Query()],
        ctx: RequestContext = Depends(get_ctx),
    ) -> api.CancelBatchPredictionJobResponse:
        log.info('Start to cancel job: %s', job_id)
        with wrap_driver_errors():
            canceled = await self._driver.cancel_batch_prediction_job_by_id(
                ctx, job_id, ProjectID(project_id))
        return api.CancelBatchPredictionJobResponse(canceled)

    async def delete_prediction_job(
            self,
            job_id: BatchPredictionJobID,
            ctx: RequestContext = Depends(get_ctx),
    ) -> EmptyResponse:
        with wrap_driver_errors():
            deleted = await self._driver.delete_batch_prediction(ctx, job_id)
        if not deleted:
            # 404 for a nonexistent job, matching the training-job DELETE.
            raise HTTPException(HTTPStatus.NOT_FOUND,
                                f'No batch prediction job found for: {job_id}')
        return EmptyResponse()

    async def update_tags(
            self,
            id: BatchPredictionJobID,
            request: Request,
            ctx: RequestContext = Depends(get_ctx),
    ) -> PatchResponse:
        job = await self.get_prediction_job(id, ctx=ctx)
        return await update_job_tags(JobType.BATCH_PREDICTION_JOB, id,
                                     job.project_id, request, self._driver)


def _require_partitioned_s3_output(
    outputs: List[api.PredictionOutputConfig]
) -> List[api.PredictionOutputConfig]:
    """Force cloud storage outputs to end with '/' so they are handled as
    directories.

    Only adds trailing slash if the path doesn't already end with a file
    extension (like .parquet or .csv), as those are meant to be files,
    not directories.
    """
    if outputs is None:
        return []
    result: List[api.PredictionOutputConfig] = []
    for output in outputs:
        if isinstance(output, (api.S3PredictionOutput, api.GCSPredictionOutput,
                               api.ADLSPredictionOutput)):
            normalized_path = output.file_path.casefold()
            if (output.file_path.startswith(
                ('s3://', 'gs://', 'abfss://', 'abfs://'))
                    and not output.file_path.endswith('/')
                    and not normalized_path.endswith(('.parquet', '.csv'))):
                output = dataclasses.replace(output,
                                             file_path=output.file_path + '/')
        result.append(output)
    return result


def _make_prediction_job_resource(
        job: BatchPredictionJobResponse) -> api.BatchPredictionJobResource:
    # Convert internal batch prediction job response to the public
    # BatchPredictionJobResource type.
    job_summary = None
    if (job.execution.status == JobStatus.DONE and job.execution.results
            and job.execution.results.output_artifact_stats):
        for stats in job.execution.results.output_artifact_stats:
            if stats.artifact_type == PredictionArtifactType.PREDICTIONS:
                job_summary = api.BatchPredictionJobSummary(
                    num_entities_predicted=stats.num_rows)

    is_running = job.execution.status == JobStatus.RUNNING
    tracking_url = (f'{get_base_kumo_url()}/'
                    f'jobs/prediction/{job.id}')
    graph_snapshot_id = None
    if job.config.graph_id and job.config.graph_data_version:
        graph_id = job.config.graph_id
        if isinstance(graph_id, GraphIDv2):
            _, fpid = GraphIDv2.parse(graph_id)
            assert fpid
            graph_snapshot_id = DataSnapshot.GRAPH.format_id(
                fpid, job.config.graph_data_version)

    public_predict_options = api.BatchPredictionOptions(
        binary_classification_threshold=(
            job.config.binary_classification_threshold),
        num_classes_to_return=job.config.num_classes_to_return,
        num_workers=job.config.num_parallel_workers_limit,
    )
    pred_table_job_id = job.config.generate_pred_table_job_id
    if job.config.prediction_time is not None and pred_table_job_id is not None:
        implicit_child_id = GeneratePredictionTableJobID.as_child_of(job.id)
        if pred_table_job_id != implicit_child_id:
            raise ValueError(
                'Prediction job has both a user-selected prediction time and '
                f'a non-child prediction-table job: {pred_table_job_id}')
        pred_table_job_id = None

    return api.BatchPredictionJobResource(
        job_id=job.id,
        project_id=job.project_id,
        config=api.BatchPredictionRequest(
            {},
            model_training_job_id=job.config.training_job_id,
            predict_options=public_predict_options,
            outputs=[
                _public_prediction_output(output)
                for output in job.config.outputs
            ],
            graph_snapshot_id=graph_snapshot_id,
            pred_table_job_id=pred_table_job_id,
            pred_table_source=job.config.custom_pred_table_source,
            prediction_time=job.config.prediction_time,
            explanations=job.config.explanations,
        ),
        job_status_report=api.JobStatusReport(
            status=job.execution.status, tracking_url=tracking_url,
            start_time=job.execution.start_time or job.timestamps.created_at,
            end_time=None if is_running else job.execution.last_updated_at,
            event_log=[job.execution.log],
            validation_response=public_job_failure_validation(
                job.execution.failure_info,
                title='Batch prediction failed',
            )),
        result=job_summary,
        created_at=job.timestamps.created_at,
        tags=job.tags,
    )
