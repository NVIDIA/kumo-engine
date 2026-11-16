# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import kumoapi.data_source as api
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rest.driver.connector_api_driver import ConnectorApiDriver
from rest.routers.connectors import ConnectorResource
from statedb.config.common import ConnectorID, IdentityID, ProjectID
from statedb.utils.errors import (
    ConnectorDoesNotExistError,
    ConnectorNameConflictError,
    IdentityForbiddenError,
)

SOURCE_PROJECT = str(ProjectID.generate())
TARGET_PROJECT = str(ProjectID.generate())


def _connector(project_id: str, name: str = 'events',
               description: str = 'raw events') -> api.ConnectorResponse:
    now = datetime.now(timezone.utc)
    return api.ConnectorResponse(
        id=str(ConnectorID.generate()),
        project_id=project_id,
        identity_id=str(IdentityID.generate()),
        name=name,
        description=description,
        timestamps=api.ResourceTimestamps(created_at=now, updated_at=now),
        config=api.S3ConnectorResourceConfig(target=api.S3RootedConnectorTarget(
            bucket='kumo-events', prefix='events', region='us-west-2')),
    )


def _driver_with_source(source: api.ConnectorResponse):
    driver = SimpleNamespace(
        _connector_service=SimpleNamespace(
            get_for_project=AsyncMock(return_value=source),
            list_for_project=AsyncMock(return_value=[]),
        ),
        create_project_connector=AsyncMock(
            side_effect=lambda project_id, args, principal: _connector(
                project_id, args.name, args.description)),
    )
    return driver


@pytest.mark.asyncio
async def test_copy_reuses_source_config_identity_and_metadata() -> None:
    source = _connector(SOURCE_PROJECT)
    driver = _driver_with_source(source)
    principal = MagicMock()

    copied = await ConnectorApiDriver.copy_project_connector(
        driver, ProjectID(SOURCE_PROJECT), ConnectorID(source.id),
        api.CopyConnectorArgs(target_project_id=TARGET_PROJECT), principal)

    driver._connector_service.get_for_project.assert_awaited_once_with(
        SOURCE_PROJECT, source.id, principal)
    target, create_args, _ = driver.create_project_connector.await_args.args
    assert target == TARGET_PROJECT
    assert create_args == api.CreateConnectorArgs(
        name=source.name,
        config=source.config,
        identity_id=source.identity_id,
        description=source.description,
    )
    assert copied.project_id == TARGET_PROJECT


@pytest.mark.asyncio
async def test_copy_applies_name_and_description_overrides() -> None:
    source = _connector(SOURCE_PROJECT)
    driver = _driver_with_source(source)

    await ConnectorApiDriver.copy_project_connector(
        driver, ProjectID(SOURCE_PROJECT), ConnectorID(source.id),
        api.CopyConnectorArgs(target_project_id=TARGET_PROJECT,
                              name='events_copy', description=''),
        MagicMock())

    create_args = driver.create_project_connector.await_args.args[1]
    assert create_args.name == 'events_copy'
    assert create_args.description == ''


@pytest.mark.asyncio
async def test_copy_into_the_source_project_is_rejected_before_lookup() -> None:
    source = _connector(SOURCE_PROJECT)
    driver = _driver_with_source(source)

    with pytest.raises(ValueError, match='must differ'):
        await ConnectorApiDriver.copy_project_connector(
            driver, ProjectID(SOURCE_PROJECT), ConnectorID(source.id),
            api.CopyConnectorArgs(target_project_id=SOURCE_PROJECT),
            MagicMock())

    driver._connector_service.get_for_project.assert_not_awaited()
    driver.create_project_connector.assert_not_awaited()


@pytest.mark.asyncio
async def test_copy_rejects_a_name_already_used_in_the_target() -> None:
    source = _connector(SOURCE_PROJECT)
    driver = _driver_with_source(source)
    driver._connector_service.list_for_project.return_value = [
        _connector(TARGET_PROJECT, name='EVENTS')
    ]

    with pytest.raises(ConnectorNameConflictError, match='already exists'):
        await ConnectorApiDriver.copy_project_connector(
            driver, ProjectID(SOURCE_PROJECT), ConnectorID(source.id),
            api.CopyConnectorArgs(target_project_id=TARGET_PROJECT),
            MagicMock())

    driver._connector_service.list_for_project.assert_awaited_once()
    driver.create_project_connector.assert_not_awaited()


def _client(copy_result) -> tuple[TestClient, MagicMock]:
    driver = MagicMock(spec=ConnectorApiDriver)
    driver.copy_project_connector = AsyncMock(side_effect=copy_result) \
        if isinstance(copy_result, Exception) else AsyncMock(
            return_value=copy_result)
    app = FastAPI()

    @app.middleware('http')
    async def add_principal(request, call_next):
        request.state.user_authorization = MagicMock()
        return await call_next(request)

    auth = SimpleNamespace(authenticate_request_header=lambda: None)
    app.include_router(ConnectorResource(auth, driver).router)
    return TestClient(app), driver


def _copy_url(connector_id: str) -> str:
    return (f'/v0/projects/{SOURCE_PROJECT}/connectors/'
            f'{connector_id}/copy')


def test_copy_route_returns_created_connector() -> None:
    created = _connector(TARGET_PROJECT)
    client, driver = _client(created)
    connector_id = str(ConnectorID.generate())

    with client:
        response = client.post(_copy_url(connector_id), json={
            'target_project_id': TARGET_PROJECT,
            'name': 'events',
        })

    assert response.status_code == HTTPStatus.OK
    assert response.json()['project_id'] == TARGET_PROJECT
    args = driver.copy_project_connector.await_args.args
    assert args[0] == SOURCE_PROJECT
    assert args[1] == connector_id
    assert args[2] == api.CopyConnectorArgs(target_project_id=TARGET_PROJECT,
                                            name='events')


@pytest.mark.parametrize('error, status', [
    (ConnectorDoesNotExistError('missing'), HTTPStatus.NOT_FOUND),
    (IdentityForbiddenError('Identity is not mapped to the project group'),
     HTTPStatus.FORBIDDEN),
    (ConnectorNameConflictError('events'), HTTPStatus.CONFLICT),
    (ValueError('Target project must differ from the source project'),
     HTTPStatus.BAD_REQUEST),
])
def test_copy_route_maps_errors(error: Exception, status: HTTPStatus) -> None:
    client, _ = _client(error)

    with client:
        response = client.post(_copy_url(str(ConnectorID.generate())),
                               json={'target_project_id': TARGET_PROJECT})

    assert response.status_code == status


def test_copy_route_rejects_unknown_fields() -> None:
    client, driver = _client(_connector(TARGET_PROJECT))

    with client:
        response = client.post(
            _copy_url(str(ConnectorID.generate())), json={
                'target_project_id': TARGET_PROJECT,
                'config': {},
            })

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    driver.copy_project_connector.assert_not_called()
