# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

from fastapi import HTTPException, Request, Response

from io_utils import logging
from kumo.validation import ValidationError, ValidationResponse
from rest.config import training_job
from rest.config.jobs import Metric, ModelEvaluationMetrics
from rest.config.status import PatchResponse
from rest.config.training_job import TrainingJobResults
from rest.deps.authentication import RequestContext
from rest.driver.pquery_api_driver import PQueryApiDriver
from rest.utils.public import wrap_driver_errors
from statedb.config.common import JobFailureInfo, JobID, JobType, ProjectID
from statedb.config.job_tag import (
    RESERVED_KEYWORDS,
    JobTagKey,
    JobTags,
    JobTagValue,
    check_custom_tag_key,
)
from statedb.utils.errors import ProjectNotFoundError

log = logging.getLogger(__name__)

_NON_TAG_QUERY_PARAMETERS = {*RESERVED_KEYWORDS, 'project_id'}
_RESERVED_QUERY_PARAMETERS = (_NON_TAG_QUERY_PARAMETERS
                              | {'model_id', 'pquery_id', 'pquery_name'})
APPLIED_TAG_FILTERS_HEADER = 'X-Applied-Job-Tag-Filters'


def public_job_failure_validation(
    failure_info: Optional[JobFailureInfo],
    *,
    title: str,
) -> Optional[ValidationResponse]:
    """Expose the curated user message, never the internal debug stack."""
    if failure_info is None:
        return None
    message = failure_info.message or title
    prefix = f'{title}. '
    if message.casefold().startswith(prefix.casefold()):
        message = message[len(prefix):]
    return ValidationResponse(errors=[
        ValidationError(
            title=title,
            message=message,
        )
    ])


def export_training_metrics(
        result: TrainingJobResults) -> ModelEvaluationMetrics:
    exported_metrics = ModelEvaluationMetrics()

    def transform(metrics: Optional[List[training_job.Metric]]) -> List[Metric]:
        if metrics is None:
            return []
        return [Metric(m.name, m.val) for m in metrics]

    exported_metrics.test_metrics = transform(result.test_metrics)
    exported_metrics.validation_metrics = transform(result.val_metrics)
    exported_metrics.training_metrics = transform(result.train_metrics)
    return exported_metrics


# Check custom job tags and raise 422 HTTP exception if any tag is invalid.
def validate_custom_job_tags(
        raw_tags: Dict[str, str],
        allow_null_value: bool = False) -> Dict[JobTagKey, JobTagValue]:
    try:
        tags = {JobTagKey(k): JobTagValue(v) for k, v in raw_tags.items()}
        for k, v in tags.items():
            check_custom_tag_key(k)
            if not allow_null_value and JobTagValue.null_or_empty(v):
                raise ValueError(
                    f'Custom job tag key={k} has null/none/empty value: {v}')
    except ValueError as ex:
        raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, str(ex)) from ex

    return tags


async def update_job_tags(
    job_type: JobType,
    job_id: JobID,
    project_id: ProjectID,
    request: Request,
    driver: PQueryApiDriver,
) -> PatchResponse:
    if len(request.query_params) == 0:
        raise HTTPException(HTTPStatus.BAD_REQUEST, 'No job tag specified.')

    raw_tags: Dict[str, str] = {}
    for k in request.query_params:
        if k == 'project_id':
            continue
        values = request.query_params.getlist(k)
        if len(values) > 1:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                f'Job tag key={k} has multiple values: {values}')
        assert len(values) == 1
        raw_tags[str(k)] = str(values[0])

    if not raw_tags:
        raise HTTPException(HTTPStatus.BAD_REQUEST, 'No job tag specified.')

    tags = validate_custom_job_tags(raw_tags, allow_null_value=True)
    with wrap_driver_errors():
        updated = await driver.update_job_tags(
            job_type,
            job_id,
            tags,
            project_id=project_id,
        )

    return PatchResponse(updated)


def get_tag_filters(
    request: Request,
    response: Optional[Response] = None,
) -> Dict[JobTagKey, JobTagValue]:
    tag_filters = {}
    for k in request.query_params:
        key = str(k)
        if key in _NON_TAG_QUERY_PARAMETERS:
            continue
        lowered = key.lower()
        if lowered != key and lowered in _RESERVED_QUERY_PARAMETERS:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                f"Unknown query parameter '{key}'; did you mean "
                f"'{lowered}'?")
        tag_filters[key] = request.query_params[k]

    try:
        filters = {JobTagKey(k): JobTagValue(v) for k, v in tag_filters.items()}
    except ValueError as ex:
        raise HTTPException(HTTPStatus.BAD_REQUEST,
                            f'Invalid query parameter: {ex}') from ex
    if filters and response is not None:
        response.headers[APPLIED_TAG_FILTERS_HEADER] = ','.join(sorted(filters))
    return filters


async def list_job_tags(
    job_type: JobType,
    request: Request,
    driver: PQueryApiDriver,
    response: Optional[Response] = None,
) -> Dict[JobID, JobTags]:
    tag_filters = get_tag_filters(request, response)
    with wrap_driver_errors():
        return await driver.list_job_tags(job_type, {**tag_filters})


# ---------------------------------------------------------------------------
# Project-level access helpers
# ---------------------------------------------------------------------------

T = TypeVar('T')


def user_can_access_project(
    ctx: RequestContext,
    project_id: str,
) -> bool:
    """Check if the current user can access a job with the given project_id.

    Admins can access every project; other principals must have the exact
    project in their request-time visible-project set.
    """
    ua = ctx.user_authorization
    if ua.is_admin or project_id in ua.project_ids:
        return True
    if ua.external_membership_unavailable:
        raise HTTPException(
            HTTPStatus.SERVICE_UNAVAILABLE,
            'External group membership is unavailable; please retry.',
        )
    return False


def filter_by_project(
    items: List[T],
    ctx: RequestContext,
    get_project_id: Callable[[T], str],
) -> List[T]:
    """Filter a list of project-owned items by visible projects."""
    if ctx.user_authorization.is_admin:
        return items
    visible = ctx.user_authorization.project_ids
    return [item for item in items if get_project_id(item) in visible]


async def resolve_requested_project(
    ctx: RequestContext,
    driver: PQueryApiDriver,
    project_id: Optional[str],
) -> Tuple[Optional[ProjectID], bool]:
    if project_id is None:
        return None, True
    with wrap_driver_errors():
        requested_project = ProjectID(project_id)
    if not user_can_access_project(ctx, requested_project):
        return None, False
    try:
        with wrap_driver_errors():
            await driver.require_project_access(ctx, requested_project)
    except ProjectNotFoundError:
        return None, False
    return requested_project, True


def require_project_id_valid(
    ctx: RequestContext,
    project_id: str,
) -> None:
    """Validate that a project_id from a create request is accessible.

    Use for create endpoints to ensure the user can create jobs in the
    specified project.
    """
    ua = ctx.user_authorization
    if not ua.is_admin and project_id not in ua.project_ids:
        if ua.external_membership_unavailable:
            raise HTTPException(
                HTTPStatus.SERVICE_UNAVAILABLE,
                'External group membership is unavailable; please retry.',
            )
        raise HTTPException(HTTPStatus.FORBIDDEN,
                            f'Project {project_id} is not accessible.')
