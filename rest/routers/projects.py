# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Projects API router.

Project endpoints (authenticated users):
  GET    /v0/projects
  POST   /v0/projects
  GET    /v0/projects/{project_id}
  PATCH  /v0/projects/{project_id}
  POST   /v0/projects/{project_id}/archive
"""
from typing import List

from fastapi import APIRouter, Depends

from rest.config.project import UpdateProjectRequest
from rest.deps.authentication import Authentication
from rest.driver.projects_driver import ProjectsDriver
from rest.utils.routers import ValidateNameResponse
from statedb.rest.project import ProjectDBEntry


class ProjectsRouter:
    def __init__(self, auth: Authentication, driver: ProjectsDriver) -> None:
        self._driver = driver
        self.router = APIRouter(
            prefix="/v0/projects",
            tags=["Projects"],
            dependencies=[Depends(auth.authenticate_request_header)],
        )

        self.router.add_api_route(
            "/",
            self.list_visible_projects,
            response_model=List[ProjectDBEntry],
            methods=["GET"],
        )
        self.router.add_api_route(
            "/by-slug/{slug}",
            self.get_project_by_slug,
            response_model=ProjectDBEntry,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/{project_id}",
            self.get_project,
            response_model=ProjectDBEntry,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/{project_id}",
            self.update_project,
            response_model=ProjectDBEntry,
            methods=["PATCH"],
        )
        self.router.add_api_route(
            "/{project_id}/archive",
            self.archive_project,
            response_model=ProjectDBEntry,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/utils/validate_name",
            self.validate_project_name,
            response_model=ValidateNameResponse,
            methods=["POST"],
        )

    async def list_visible_projects(self, ) -> List[ProjectDBEntry]:
        return await self._driver._list_visible_projects()

    async def get_project(
        self,
        project_id: str,
    ) -> ProjectDBEntry:
        return await self._driver._get_project(project_id)

    async def get_project_by_slug(self, slug: str) -> ProjectDBEntry:
        return await self._driver.get_project_by_slug(slug)

    async def update_project(
        self,
        project_id: str,
        args: UpdateProjectRequest,
    ) -> ProjectDBEntry:
        return await self._driver._update_project(
            project_id=project_id,
            name=args.name,
            description=args.description,
            is_private=args.is_private,
        )

    async def archive_project(
        self,
        project_id: str,
    ) -> ProjectDBEntry:
        return await self._driver._archive_project(project_id)

    async def validate_project_name(
        self,
        name: str,
    ) -> ValidateNameResponse:
        """Validate that a normalized project name is globally available."""
        return await self._driver.validate_project_name(name=name)
