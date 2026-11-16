# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Any, Callable, Optional, TypeVar

from fastapi import HTTPException

ResourceT = TypeVar("ResourceT")


def require_project_id(project_id: Optional[str]) -> str:
    """Require the project ownership coordinate on every resource operation."""
    if project_id is None:
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            "project_id is required",
        )
    return project_id


def require_visible_project_resource(
    resource: Optional[ResourceT],
    *,
    requested_project_id: str,
    not_found_error_factory: Callable[[], Exception],
) -> ResourceT:
    """Hide absent and cross-project resources behind the same not-found error."""
    if resource is None:
        raise not_found_error_factory()
    resource_project_id = getattr(resource, "project_id", None)
    if resource_project_id is None or str(resource_project_id) != str(
            requested_project_id):
        raise not_found_error_factory()
    return resource


def require_visible_project_resource_for_user(
    resource: Optional[ResourceT],
    *,
    requested_project_id: str,
    user_authorization: Any,
    not_found_error_factory: Callable[[], Exception],
) -> ResourceT:
    resource = require_visible_project_resource(
        resource,
        requested_project_id=requested_project_id,
        not_found_error_factory=not_found_error_factory,
    )
    if getattr(user_authorization, "is_admin", False):
        return resource
    if requested_project_id not in getattr(user_authorization, "project_ids",
                                           ()):
        if getattr(user_authorization, "external_membership_unavailable",
                   False):
            raise HTTPException(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "External group membership is unavailable; please retry.",
            )
        raise not_found_error_factory()
    return resource


def require_requested_project_visible_for_user(
    requested_project_id: str,
    *,
    user_authorization: Any,
    not_found_error_factory: Callable[[], Exception],
) -> None:
    if getattr(user_authorization, "is_admin", False):
        return
    if requested_project_id not in getattr(user_authorization, "project_ids",
                                           ()):
        if getattr(user_authorization, "external_membership_unavailable",
                   False):
            raise HTTPException(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "External group membership is unavailable; please retry.",
            )
        raise not_found_error_factory()
