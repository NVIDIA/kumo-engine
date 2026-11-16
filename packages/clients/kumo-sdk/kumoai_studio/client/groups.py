# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from kumoai_studio.client.endpoints import GroupEndpoints
from kumoai_studio.client.transport import Transport
from kumoai_studio.client.utils import raise_on_error, response_json
from kumoai_studio.project import Group


class GroupsAPI:
    r"""Typed API for native Kumo group discovery."""
    def __init__(self, client: Transport) -> None:
        self._client = client

    @staticmethod
    def _from_json(data: dict) -> Group:
        return Group(id=data['id'], slug=data['slug'],
                     name=data['display_name'])

    def list_groups(self) -> List[Group]:
        r"""List all native Kumo groups visible to the authenticated user.

        Returns:
            Credential-free :class:`~kumoai_studio.project.Group` values.
        """
        resp = self._client._request(GroupEndpoints.list)
        raise_on_error(resp)
        return [self._from_json(group) for group in response_json(resp)]

    def get_group_by_slug(self, slug: str) -> Group:
        r"""Get one visible native group by exact canonical slug."""
        resp = self._client._request(GroupEndpoints.by_slug(slug))
        raise_on_error(resp)
        return self._from_json(response_json(resp))
