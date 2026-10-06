# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest

from rest.config.group import CreateKumoGroupRequest
from rest.driver.groups_driver import GroupsDriver
from statedb.config.common import GroupID
from statedb.config.timestamp import ResourceTimestamps
from statedb.rest.group import KumoGroupDBEntry


def _timestamps() -> ResourceTimestamps:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ResourceTimestamps(created_at=now, updated_at=now)


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


class _GroupService:
    def __init__(self) -> None:
        self.calls = []
        self.group_id = GroupID(f"group_{'a' * 32}")

    async def list_kumo_groups(self, principal):
        self.calls.append(("list_kumo_groups", ))
        return [
            KumoGroupDBEntry(
                id=self.group_id,
                slug="engineering",
                display_name="Engineering",
                normalized_display_name="engineering",
                timestamps=_timestamps(),
            )
        ]

    async def get_kumo_group_by_alias(self, display_name, user_authorization):
        self.calls.append(
            ("get_kumo_group_by_alias", display_name, user_authorization))
        return KumoGroupDBEntry(
            id=self.group_id,
            slug="engineering",
            display_name=display_name,
            normalized_display_name=display_name.casefold(),
            timestamps=_timestamps(),
        )

    async def create_kumo_groups(self, groups, principal):
        self.calls.append(("create_kumo_groups", groups))
        return [
            KumoGroupDBEntry(
                id=self.group_id,
                slug=groups[0].slug or "engineering",
                display_name=groups[0].display_name,
                normalized_display_name=groups[0].display_name.casefold(),
                timestamps=_timestamps(),
            )
        ]

    async def update_kumo_group(self, group_id, display_name, description, *,
                                principal):
        self.calls.append(
            ("update_kumo_group", group_id, display_name, description))
        return KumoGroupDBEntry(
            id=group_id,
            slug="engineering",
            display_name=display_name or "Engineering",
            normalized_display_name=(display_name or "Engineering").casefold(),
            description=description or "",
            timestamps=_timestamps(),
        )

    async def delete_kumo_group(self, group_id, principal):
        self.calls.append(("delete_kumo_group", group_id))

    async def list_user_groups(self, user_authorization):
        self.calls.append(("list_user_groups", user_authorization))
        return [
            KumoGroupDBEntry(
                id=self.group_id,
                slug="engineering",
                display_name="Engineering",
                normalized_display_name="engineering",
                timestamps=_timestamps(),
            )
        ]


def _driver(group_service: _GroupService):
    state_services = type("_StateServices", (), {"groups": group_service})()
    driver = GroupsDriver()
    driver._configure_state_services(state_services)
    return driver


@pytest.mark.asyncio
async def test_groups_driver_uses_state_services_for_group_operations():
    service = _GroupService()
    driver = _driver(service)
    group_id = service.group_id
    create_request = CreateKumoGroupRequest(display_name="Engineering", )

    assert await driver._list_kumo_groups()
    assert (await driver._get_kumo_group_by_alias("Engineering")).id == group_id
    assert await driver._create_kumo_groups([create_request])
    assert (await driver._update_kumo_group(
        group_id,
        display_name="Eng",
        description="Updated",
    )).display_name == "Eng"
    await driver._delete_kumo_group(str(group_id))
    assert await driver._list_user_groups()

    assert [call[0] for call in service.calls] == [
        "list_kumo_groups",
        "get_kumo_group_by_alias",
        "create_kumo_groups",
        "update_kumo_group",
        "delete_kumo_group",
        "list_user_groups",
    ]
