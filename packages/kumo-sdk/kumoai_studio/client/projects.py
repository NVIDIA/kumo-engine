# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from kumoapi.data_source import CreateProjectArgs
from kumoapi.json_serde import to_json_dict

from kumoai_studio.client.endpoints import ProjectEndpoints
from kumoai_studio.client.transport import Transport
from kumoai_studio.client.utils import raise_on_error, response_json
from kumoai_studio.project import Project


class ProjectsAPI:
    r"""Typed API for project operations."""
    def __init__(self, client: Transport) -> None:
        self._client = client

    @staticmethod
    def _from_json(data: dict) -> Project:
        return Project(
            id=data['id'],
            slug=data['slug'],
            name=data['display_name'],
            group_id=data['group_id'],
        )

    def list_projects(self, group_id: Optional[str] = None) -> List[Project]:
        r"""List visible projects, optionally narrowed to one group.

        Args:
            group_id: Immutable group ID to filter by. ``None`` returns every
                project visible to the authenticated user.

        Returns:
            Credential-free :class:`~kumoai_studio.project.Project` values.
        """
        params = {'group_id': group_id} if group_id is not None else None
        resp = self._client._request(
            ProjectEndpoints.list,
            params=params,
        )
        raise_on_error(resp)
        return [self._from_json(project) for project in response_json(resp)]

    def get_project(self, project_id: str) -> Project:
        r"""Get one visible project by immutable ID.

        Args:
            project_id: Immutable project ID.

        Returns:
            The visible :class:`~kumoai_studio.project.Project`.
        """
        resp = self._client._request(ProjectEndpoints.get(project_id))
        raise_on_error(resp)
        return self._from_json(response_json(resp))

    def get_project_by_slug(self, slug: str) -> Project:
        r"""Get one visible project by its exact canonical slug."""
        resp = self._client._request(ProjectEndpoints.by_slug(slug))
        raise_on_error(resp)
        return self._from_json(response_json(resp))

    def create_project(self, name: str, group_id: str,
                       is_private: bool = False,
                       slug: Optional[str] = None) -> Project:
        r"""Create a project within a native Kumo group.

        Args:
            name: Globally unique project display name.
            group_id: Immutable ID of a group visible to the caller.
            is_private: Whether only the creator, rather than all group
                members, can access the project.
            slug: Optional immutable short ID. The server derives it once from
                ``name`` when omitted.

        Returns:
            The newly created :class:`~kumoai_studio.project.Project`.
        """
        args = CreateProjectArgs(name=name, group_id=group_id, slug=slug,
                                 is_private=is_private)
        payload = to_json_dict(args)
        if slug is None:
            payload.pop('slug', None)
        resp = self._client._request(ProjectEndpoints.create, json=payload)
        raise_on_error(resp)
        return self._from_json(response_json(resp))
