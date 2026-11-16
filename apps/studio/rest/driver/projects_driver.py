# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional

from rest.deps.context import _current_user_ctx
from rest.utils.routers import ValidateNameResponse
from statedb.db.services import ProjectService, StateServices
from statedb.rest.project import ProjectDBEntry

log = logging.getLogger(__name__)


class ProjectsDriver:
    def __init__(self):
        self._state_services: Optional[StateServices] = None

    def _configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services

    @property
    def _project_service(self) -> ProjectService:
        if self._state_services is None:
            raise RuntimeError("ProjectsDriver StateServices is not configured")
        return self._state_services.projects

    # ========================= Project CRUD =========================

    async def create_project(
        self,
        name: str,
        group_id: str,
        slug: Optional[str] = None,
        description: str = "",
        is_private: bool = False,
    ) -> ProjectDBEntry:
        user_auth = _current_user_ctx.get()
        if user_auth is None:
            # raising runtime error because this should never really happen in prod
            raise RuntimeError(
                "create_project requires an authenticated user context")
        return await self._project_service.create_project(
            name=name,
            group_id=group_id,
            slug=slug,
            description=description,
            is_private=is_private,
            user_authorization=user_auth,
        )

    async def _list_visible_projects(self) -> List[ProjectDBEntry]:
        return await self._project_service.list_visible_projects(
            _current_user_ctx.get())

    async def list_projects_in_group(self,
                                     group_id: str) -> List[ProjectDBEntry]:
        return await self._project_service.list_projects_in_group(
            group_id,
            _current_user_ctx.get(),
        )

    async def _get_project(self, project_id: str) -> ProjectDBEntry:
        return await self._project_service.get_project(
            project_id,
            _current_user_ctx.get(),
        )

    async def get_project_by_slug(self, slug: str) -> ProjectDBEntry:
        return await self._project_service.get_project_by_slug(
            slug,
            _current_user_ctx.get(),
        )

    async def _archive_project(self, project_id: str) -> ProjectDBEntry:
        return await self._project_service.archive_project(
            project_id,
            _current_user_ctx.get(),
        )

    async def _update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_private: Optional[bool] = None,
    ) -> ProjectDBEntry:
        return await self._project_service.update_project(
            project_id=project_id,
            name=name,
            description=description,
            is_private=is_private,
            user_authorization=_current_user_ctx.get(),
        )

    # ========================= Project resource threading =========================

    async def validate_project_name(
        self,
        name: str,
        exclude_project_id: Optional[str] = None,
    ) -> ValidateNameResponse:
        """Validate that a normalized project display name is globally unique.

        Args:
            name: The project display name to validate.
            exclude_project_id: Optional project ID to exclude from the uniqueness
                check (used when renaming an existing project).

        Returns:
            ValidateNameResponse with is_valid=True if the name is available.
        """

        return await self._project_service.validate_project_name(
            name=name,
            exclude_project_id=exclude_project_id,
            user_authorization=_current_user_ctx.get(),
        )

    async def _validate_project_for_creation(
        self,
        project_id: str,
    ) -> ProjectDBEntry:
        return await self._project_service.validate_project_for_creation(
            project_id,
            _current_user_ctx.get(),
        )
