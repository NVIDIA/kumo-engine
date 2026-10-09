# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
from datetime import timedelta
from typing import Union

from kumoapi.data_source import ConnectorType
from kumoapi.jobs import (
    FileTrainingTableOutputConfig,
    NamedTrainingTableOutputConfig,
    PredictionOutputConfig,
    TrainingTableOutputConfig,
)

from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.data_backend import DataBackend
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.connector import normalize_file_destination
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.prediction_output import (
    ADLSPredictionOutput,
    GCSPredictionOutput,
    S3PredictionOutput,
    normalize_object_prediction_output,
)
from rest.deps.context import _current_user_ctx
from statedb.config.common import (
    ArtifactExportJobID,
    ConnectorID,
    JobID,
    JobStatus,
    ProjectID,
)
from statedb.config.timestamp import now_utc
from statedb.db.connector_authorizations import canonical_request_digest
from statedb.db.services import StateServices
from statedb.rest.connector_authorization import CONNECTOR_OUTPUT_ACTIONS

_ARTIFACT_UNSEALED_DISPATCH_WINDOW = timedelta(hours=1)

log = logging.getLogger(__name__)


def _artifact_export_job_id(
    *,
    user_id: str,
    project_id: ProjectID,
    request_id: str,
) -> ArtifactExportJobID:
    if not request_id:
        raise ValueError('Artifact export request_id is required')
    value = f'artifact-export:v1:{user_id}:{project_id}:{request_id}'
    digest = hashlib.blake2b(value.encode('utf-8'), digest_size=16).hexdigest()
    return ArtifactExportJobID(f'ae-job-{digest}')


class ArtifactApiDriver:
    def __init__(
        self,
        ml_job_backend: MLJobBackend,
        data_backend_ctx: DataBackendContext,
    ):
        self._ml_job_backend = ml_job_backend
        self._data_backend_ctx = data_backend_ctx
        self._state_services: StateServices | None = None

    def _configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services

    @property
    def _required_state_services(self) -> StateServices:
        if self._state_services is None:
            raise RuntimeError(
                "ArtifactApiDriver StateServices is not configured")
        return self._state_services

    async def _get_execution_connector(
        self,
        connector_id: ConnectorID,
        project_id: ProjectID,
    ) -> ConnectorExecutionRef:
        connector_service = self._required_state_services.connectors
        return await connector_service.execution_ref_for_project(
            _current_user_ctx.get(), project_id, connector_id,
            CONNECTOR_OUTPUT_ACTIONS)

    async def create_artifact_export_job(
        self,
        job_id: str,
        output_config: Union[PredictionOutputConfig, TrainingTableOutputConfig],
        project_id: ProjectID,
        request_id: str,
    ) -> str:
        principal = _current_user_ctx.get()
        if principal is None:
            raise PermissionError('Authentication is required')
        await self._required_state_services.connectors.require_project_access(
            principal, project_id)
        workflow_id = _artifact_export_job_id(
            user_id=str(principal.user_id),
            project_id=project_id,
            request_id=request_id,
        )
        request_digest = canonical_request_digest({
            'job_id':
            job_id,
            'output_config':
            output_config,
        })

        async def close_existing_request(entry) -> bool:
            if entry.dispatch_sealed:
                return True
            status = await DataBackend.get_temporal_job_status(
                self._data_backend_ctx, str(workflow_id))
            if (status != JobStatus.MISSING
                    or now_utc() - entry.timestamps.created_at
                    > _ARTIFACT_UNSEALED_DISPATCH_WINDOW):
                await self._required_state_services.artifact_exports.seal_dispatch(
                    workflow_id, project_id=project_id)
                return True
            return False

        artifact_exports = self._required_state_services.artifact_exports
        entry = await artifact_exports.get(workflow_id)
        if entry is not None:
            entry, _ = await artifact_exports.create(
                id=workflow_id,
                project_id=project_id,
                created_by=principal.user_id,
                request_id=request_id,
                request_digest=request_digest,
            )
            if await close_existing_request(entry):
                return str(workflow_id)

        await self._require_source_jobs_in_project(job_id, output_config,
                                                   project_id)

        output = output_config
        if not hasattr(output, 'connector_id'):
            raise ValueError(
                'Artifact exports require a project-owned connector')
        conn_id = ConnectorID(output.connector_id)
        connector = await self._required_state_services.connectors.get_for_project(
            project_id, conn_id, principal)
        if connector.config.type == ConnectorType.DATABRICKS:
            raise ValueError(
                'Post-hoc Databricks artifact export is not supported. '
                'Prediction outputs must be configured when creating the '
                'batch prediction job so GPU workers upload Parquet directly '
                'to the connector-managed Unity Catalog volume.')
        if isinstance(output_config, (
                NamedTrainingTableOutputConfig,
                FileTrainingTableOutputConfig,
        )):
            connector_type = connector.config.type
            if (isinstance(output_config, NamedTrainingTableOutputConfig)
                    and connector_type not in {
                        ConnectorType.SNOWFLAKE,
                        ConnectorType.BIGQUERY,
                    }):
                raise ValueError(
                    'Named training-table output requires a warehouse '
                    'connector')
            if (isinstance(output_config, FileTrainingTableOutputConfig)
                    and connector_type not in {
                        ConnectorType.S3,
                        ConnectorType.GCS,
                        ConnectorType.ADLS,
                    }):
                raise ValueError(
                    'File training-table output requires an object-store '
                    'connector')
            if isinstance(output_config, FileTrainingTableOutputConfig):
                output_config.path = normalize_file_destination(
                    connector.config, output_config.path)
        else:
            expected_type = ConnectorType[output_config.storage_type.name]
            if connector.config.type != expected_type:
                raise ValueError(
                    f'{expected_type.value} prediction output requires an '
                    f'{expected_type.value} connector')
            if isinstance(output_config,
                          (S3PredictionOutput, GCSPredictionOutput,
                           ADLSPredictionOutput)):
                normalize_object_prediction_output(output_config,
                                                   connector.config)

        if entry is None:
            entry, inserted = await artifact_exports.create(
                id=workflow_id,
                project_id=project_id,
                created_by=principal.user_id,
                request_id=request_id,
                request_digest=request_digest,
            )
            if not inserted and await close_existing_request(entry):
                return str(workflow_id)
        connector_spec = await self._get_execution_connector(
            conn_id, project_id)

        # The seal is an admission barrier, not a lock held across Temporal.
        # A cancellation that seals before this final check blocks dispatch;
        # one that races after it is handled best-effort after the start.
        admitted_entry = await artifact_exports.get(workflow_id)
        if admitted_entry is None:
            raise RuntimeError('Artifact export request disappeared')
        if admitted_entry.dispatch_sealed:
            return str(workflow_id)

        started_id = await DataBackend.create_artifact_export_job(
            self._data_backend_ctx,
            job_id,
            output_config,
            connector_spec,
            workflow_id,
        )
        sealed_entry = await artifact_exports.seal_dispatch(
            workflow_id,
            project_id=project_id,
        )
        if sealed_entry.cancel_requested:
            try:
                await DataBackend.cancel_temporal_job(self._data_backend_ctx,
                                                      str(started_id))
            except BaseException:
                log.exception('Failed best-effort cancellation of artifact '
                              'dispatch that raced with cancellation')
        return str(started_id)

    async def get_artifact_export_job_status(
        self,
        job_id: str,
        project_id: ProjectID,
    ) -> JobStatus:
        await self._required_state_services.artifact_exports.get_for_project(
            ArtifactExportJobID(job_id), project_id, _current_user_ctx.get())
        return await DataBackend.get_temporal_job_status(
            self._data_backend_ctx, job_id)

    async def cancel_artifact_export_job(
        self,
        job_id: str,
        project_id: ProjectID,
    ) -> JobStatus:
        artifact_exports = self._required_state_services.artifact_exports
        await artifact_exports.get_for_project(ArtifactExportJobID(job_id),
                                               project_id,
                                               _current_user_ctx.get())
        await artifact_exports.request_cancel(ArtifactExportJobID(job_id),
                                              project_id=project_id)
        status = await DataBackend.cancel_temporal_job(self._data_backend_ctx,
                                                       job_id)
        return status

    async def _require_source_jobs_in_project(
        self,
        job_id: str,
        output_config: Union[PredictionOutputConfig, TrainingTableOutputConfig],
        project_id: ProjectID,
    ) -> tuple[str, ...]:
        source_ids = (job_id, )
        if any(not source_id for source_id in source_ids):
            raise ValueError('Artifact export source job is required')
        for source_id in source_ids:
            source_job = await self._required_state_services.ml_jobs.get_job_if_exists_generic(
                JobID(source_id))
            if source_job is None or source_job.project_id != project_id:
                raise ValueError(
                    'Artifact export source job is not in the requested '
                    'project')
        return tuple(source_ids)
