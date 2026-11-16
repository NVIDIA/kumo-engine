# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List

from fastapi import APIRouter, Depends
from kumoapi import data_source as api

from rest.config.group import ListUserKumoGroupsResponse
from rest.deps.authentication import Authentication
from rest.driver.groups_driver import GroupsDriver
from rest.utils.public import wrap_driver_errors


class GroupsRouter:
    def __init__(
        self,
        api_key_auth: Callable,
        groups_driver: GroupsDriver,
        auth: Authentication,
    ) -> None:
        self._driver = groups_driver
        self._auth = auth

        self.router = APIRouter(
            prefix="/groups",
            tags=["API v1 (Under Development): Groups"],
            dependencies=[Depends(api_key_auth)],
        )

        self.router.add_api_route(
            "/by-slug/{slug}",
            self.get_group_by_slug,
            response_model=api.GroupResponse,
            methods=['GET'],
            summary="Get one visible group by immutable slug.",
        )
        self.router.add_api_route(
            "",
            self.list_user_groups,
            response_model=List[api.GroupResponse],
            methods=['GET'],
            summary="List groups the current user belongs to.",
        )

    async def list_user_groups(self) -> List[api.GroupResponse]:
        with wrap_driver_errors():
            groups: List[ListUserKumoGroupsResponse] = (
                await self._driver.list_user_groups())
            return [
                api.GroupResponse(
                    id=group.group_id,
                    slug=group.slug,
                    display_name=group.display_name,
                ) for group in groups
            ]

    async def get_group_by_slug(self, slug: str) -> api.GroupResponse:
        with wrap_driver_errors():
            group = await self._driver.get_kumo_group_by_slug(slug)
            return api.GroupResponse(
                id=group.id,
                slug=group.slug,
                display_name=group.display_name,
            )
