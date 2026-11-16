# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional

from fastapi import APIRouter, Depends
from kumoapi.data_source import CreateProjectArgs, ProjectResponse

from rest.deps.authentication import Authentication
from rest.driver.projects_driver import ProjectsDriver
from rest.utils.public import wrap_driver_errors
from statedb.config.common import GroupID
from statedb.rest.project import ProjectDBEntry


def _public_project(project: ProjectDBEntry) -> ProjectResponse:
    r"""Project persisted metadata onto the public project allowlist."""
    return ProjectResponse(
        id=project.id,
        slug=project.slug,
        display_name=project.display_name,
        group_id=project.group_id,
    )


class ProjectsRouter:
    def __init__(
        self,
        api_key_auth: Callable,
        projects_driver: ProjectsDriver,
        auth: Authentication,
    ) -> None:
        self._driver = projects_driver
        self._auth = auth

        self.router = APIRouter(
            prefix="/projects",
            tags=["API v1 (Under Development): Projects"],
            dependencies=[Depends(api_key_auth)],
        )

        self.router.add_api_route(
            "",
            self.list_projects,
            response_model=List[ProjectResponse],
            methods=['GET'],
            summary="List projects in a group.",
        )
        self.router.add_api_route(
            "/by-slug/{slug}",
            self.get_project_by_slug,
            response_model=ProjectResponse,
            methods=['GET'],
            summary="Get one visible project by immutable slug.",
        )
        self.router.add_api_route(
            "/{project_id}",
            self.get_project,
            response_model=ProjectResponse,
            methods=['GET'],
            summary="Get one visible project by immutable ID.",
        )

        self.router.add_api_route(
            "",
            self.create_project,
            response_model=ProjectResponse,
            methods=['POST'],
            summary="Create a new project.",
        )

    async def list_projects(
        self,
        group_id: Optional[str] = None,
    ) -> List[ProjectResponse]:
        with wrap_driver_errors():
            if group_id is None:
                projects = await self._driver._list_visible_projects()
            else:
                projects = await self._driver.list_projects_in_group(
                    group_id=GroupID(group_id))
            return [_public_project(project) for project in projects]

    async def get_project(self, project_id: str) -> ProjectResponse:
        with wrap_driver_errors():
            return _public_project(await self._driver._get_project(project_id))

    async def get_project_by_slug(self, slug: str) -> ProjectResponse:
        with wrap_driver_errors():
            return _public_project(await self._driver.get_project_by_slug(slug))

    async def create_project(
        self,
        request: CreateProjectArgs,
    ) -> ProjectResponse:
        with wrap_driver_errors():
            project = await self._driver.create_project(
                name=request.name,
                group_id=request.group_id,
                slug=request.slug,
                description=request.description,
                is_private=request.is_private,
            )
            return _public_project(project)
