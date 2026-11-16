# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent
import concurrent.futures
import time
import uuid
from typing import Literal, Optional, Union, overload

from kumoapi.common import JobStatus
from kumoapi.jobs import ArtifactExportRequest
from kumoapi.plugin import PublishedModel
from typing_extensions import override

from kumoai_studio import global_state
from kumoai_studio.artifact_export.config import ModelExportConfig
from kumoai_studio.backend import cloud_backend
from kumoai_studio.futures import KumoProgressFuture, create_future


def _resolve_project_id(project_id: Optional[str]) -> str:
    if project_id is not None:
        return project_id
    from kumoai_studio import _require_project
    _require_project()
    assert global_state._project_id is not None
    return global_state._project_id


class ArtifactExportResult:
    r"""Represents a completed artifact export job.

    Args:
        job_id: ID of the completed artifact export job.
        project_id: Exact project that owns the job. ``None`` snapshots the
            active project.
    """
    def __init__(self, job_id: str, project_id: Optional[str] = None) -> None:
        self.job_id = job_id
        self._kumo_project_id = _resolve_project_id(project_id)

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable Kumo project owning this export result."""
        return self._kumo_project_id

    @property
    def published_model(self) -> Optional[PublishedModel]:
        """Return the model this export published, if it published one.

        ``None`` for a prediction or training-table export, and for a model
        written to a path, which the caller already named in full. A registry
        assigns the version when the caller named none, so this is the only
        way to learn what was created.
        """
        api = cloud_backend().artifact_export
        return api.published_model(self.job_id,
                                   project_id=self._kumo_project_id)

    def tracking_url(self) -> str:
        r"""Returns a tracking URL pointing to the UI display of
        this prediction export job.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(job_id={self.job_id})"


class ArtifactExportJob(KumoProgressFuture[ArtifactExportResult]):
    """Represents an in-progress artifact export job.

    Args:
        job_id: ID of the artifact export job.
        project_id: Exact project that owns the job. ``None`` snapshots the
            active project. Later active-project changes do not redirect the
            job.
    """
    def __init__(self, job_id: str, project_id: Optional[str] = None) -> None:
        self.job_id = job_id
        self._kumo_project_id = _resolve_project_id(project_id)
        self._fut: concurrent.futures.Future = create_future(
            _poll_export(job_id, self._kumo_project_id))

    @property
    def id(self) -> str:
        """The unique ID of this export job."""
        return self.job_id

    @property
    def kumo_project_id(self) -> str:
        """Return the immutable Kumo project owning this export job."""
        return self._kumo_project_id

    @override
    def result(self) -> ArtifactExportResult:
        return self._fut.result()

    @override
    def future(self) -> 'concurrent.futures.Future[ArtifactExportResult]':
        return self._fut

    @override
    def _attach_internal(
        self,
        interval_s: float = 20.0,
    ) -> ArtifactExportResult:
        """Allows a user to attach to a running export job and view
        its progress.

        Args:
            interval_s (float): Time interval (seconds) between polls, minimum
                value allowed is 4 seconds.
        """
        assert interval_s >= 4.0
        print(f"Attaching to export job {self.job_id}. To detach from "
              f"this job, please enter Ctrl+C (the job will continue to run, "
              f"and you can re-attach anytime).")

        # TODO improve print statements.
        # Will require changes to status to return
        # JobStatusReport instead of JobStatus.
        while not self.done():
            status = self.status()
            print(f"Export job {self.job_id} status: {status}")
            time.sleep(interval_s)

        return self.result()

    def status(self) -> JobStatus:
        """Returns the status of a running export job."""
        return get_export_status(self.job_id, self._kumo_project_id)

    def cancel(self) -> bool:
        """Cancels a running export job.

        Returns:
            bool: True if the job is in a terminal state.
        """
        api = cloud_backend().artifact_export
        status = api.cancel(self.job_id, project_id=self._kumo_project_id)
        if status == JobStatus.CANCELLED:
            return True
        return False


@overload
def export_model(
    config: ModelExportConfig,
    *,
    request_id: Optional[str] = ...,
    project_id: Optional[str] = ...,
) -> 'ArtifactExportJob':
    pass


@overload
def export_model(
    config: ModelExportConfig,
    *,
    non_blocking: Literal[True],
    request_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> 'ArtifactExportJob':
    pass


@overload
def export_model(
    config: ModelExportConfig,
    *,
    non_blocking: Literal[False],
    request_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> 'ArtifactExportResult':
    pass


@overload
def export_model(
    config: ModelExportConfig,
    *,
    non_blocking: bool,
    request_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Union['ArtifactExportJob', 'ArtifactExportResult']:
    pass


def export_model(
    config: ModelExportConfig,
    *,
    non_blocking: bool = True,
    request_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Union['ArtifactExportJob', 'ArtifactExportResult']:
    r"""Export an offline model bundle to an external storage location.

    Args:
        config: A :class:`~kumoai_studio.artifact_export.ModelExportConfig` with the
            training job, SDK output connector, and destination.
        non_blocking: If ``True``, returns an
            :class:`~kumoai_studio.artifact_export.ArtifactExportJob` future. If
            ``False``, blocks until complete and returns an
            :class:`~kumoai_studio.artifact_export.ArtifactExportResult`.
        request_id: Canonical UUID for idempotent submission. ``None``
            generates one. Reuse the same explicit value when retrying the
            same export request.
        project_id: Exact owning Kumo project. ``None`` uses the output
            connector's immutable project. An explicit value must match the
            connector and never changes global SDK state.
    """
    connector_project_id = config.kumo_project_id
    if project_id is None:
        project_id = connector_project_id
    elif project_id != connector_project_id:
        raise ValueError(
            f'Output connector belongs to project {connector_project_id!r}, '
            f'not {project_id!r}.')
    api = cloud_backend().artifact_export
    # Refused here rather than on the wire: the Kumo service does not publish
    # to a registry, and a caller who asked for one deserves to be told that
    # instead of a rejection from somewhere inside the request.
    # `is True` rather than truthiness: an area that turns unknown attributes
    # into raisers -- which several do, to report a gap by name -- would hand
    # back a callable here, and a callable is truthy.
    if config.is_registry and getattr(api, 'supports_model_registry',
                                      False) is not True:
        raise NotImplementedError(
            f"The {cloud_backend().name} backend cannot publish a model to a "
            f"registry, so output_location has nowhere to go. Export to an "
            f"S3, GCS, or ADLS connector with output_path instead of "
            f"{type(config.output_connector).__name__}.")
    submission_id = str(uuid.uuid4()) if request_id is None else request_id
    api_config = config._to_api_config()
    job_id = api.create(
        ArtifactExportRequest(request_id=submission_id,
                              job_id=api_config.training_job_id,
                              model_output=api_config), project_id=project_id)
    if non_blocking:
        return ArtifactExportJob(job_id, project_id=project_id)
    return ArtifactExportJob(job_id, project_id=project_id).attach()


def get_export_status(job_id: str,
                      project_id: Optional[str] = None) -> JobStatus:
    """Return the current status of a project-owned artifact export job.

    Args:
        job_id: ID of the artifact export job.
        project_id: Exact owning project. ``None`` uses the active project.
    """
    api = cloud_backend().artifact_export
    resource = api.get(job_id, project_id=project_id)
    return resource


async def _poll_export(
        job_id: str, project_id: Optional[str] = None) -> ArtifactExportResult:
    status = get_export_status(job_id, project_id)
    while not status.is_terminal and status != JobStatus.MISSING:
        await asyncio.sleep(10)
        status = get_export_status(job_id, project_id)

    if status == JobStatus.MISSING:
        raise RuntimeError(
            f"Export job {job_id} is missing. Its workflow history may have "
            "expired.")

    if status != JobStatus.DONE:
        raise RuntimeError(f"Export job {job_id} failed "
                           f"with job status {status}.")

    return ArtifactExportResult(job_id=job_id, project_id=project_id)
