# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import kumoapi.data_source as api
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from rest.driver.connector_api_driver import ConnectorApiDriver
from rest.routers.identities import IdentitiesResource
from statedb.config.common import ConnectorID, IdentityID, ProjectID


def _auth(*, admin: bool = True):
    def require_customer_admin() -> None:
        if not admin:
            raise HTTPException(HTTPStatus.FORBIDDEN,
                                'Administrator access required')

    return SimpleNamespace(
        authenticate_request_header=lambda: None,
        require_customer_admin=require_customer_admin,
    )


def _app_with_service(service, *, admin: bool = True) -> FastAPI:
    driver = MagicMock(spec=ConnectorApiDriver)
    driver._require_state_services.return_value = SimpleNamespace(
        identities=service)
    app = FastAPI()

    @app.middleware('http')
    async def add_principal(request, call_next):
        request.state.user_authorization = MagicMock()
        return await call_next(request)

    app.include_router(IdentitiesResource(_auth(admin=admin), driver).router)
    return app


def test_list_connectors_requires_customer_admin() -> None:
    identity_id = str(IdentityID.generate())
    service = SimpleNamespace(list_connectors_using_identity=AsyncMock())
    app = _app_with_service(service, admin=False)

    with TestClient(app) as client:
        response = client.get(f'/v0/identities/{identity_id}/connectors')

    assert response.status_code == HTTPStatus.FORBIDDEN
    service.list_connectors_using_identity.assert_not_called()


def test_list_connectors_returns_blocking_connectors() -> None:
    identity_id = str(IdentityID.generate())
    connector_id = str(ConnectorID.generate())
    active_project = str(ProjectID.generate())
    missing_project = str(ProjectID.generate())
    resp = api.IdentityConnectorsResponse(
        identity_id=identity_id,
        connectors=[
            api.IdentityConnectorUsage(
                connector_id=connector_id,
                connector_name='Data Warehouse',
                project_id=active_project,
                project_display_name='Sales',
            ),
            api.IdentityConnectorUsage(
                connector_id=str(ConnectorID.generate()),
                connector_name='Orphan',
                project_id=missing_project,
                project_display_name=None,
            ),
        ],
    )
    service = SimpleNamespace(
        list_connectors_using_identity=AsyncMock(return_value=resp))
    app = _app_with_service(service)

    with TestClient(app) as client:
        response = client.get(f'/v0/identities/{identity_id}/connectors')

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        'identity_id': identity_id,
        'connectors': [
            {
                'connector_id': connector_id,
                'connector_name': 'Data Warehouse',
                'project_id': active_project,
                'project_display_name': 'Sales',
            },
            {
                'connector_id': resp.connectors[1].connector_id,
                'connector_name': 'Orphan',
                'project_id': missing_project,
                'project_display_name': None,
            },
        ],
    }
    service.list_connectors_using_identity.assert_awaited_once()
