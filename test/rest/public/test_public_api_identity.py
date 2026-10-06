# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import kumoapi.data_source as api
import pytest

from rest.driver.connector_api_driver import ConnectorApiDriver
from rest.public.routers.identity import IdentityRouter
from rest.routers.identities import IdentitiesResource
from statedb.config.common import IdentityID, ProjectID


@pytest.mark.asyncio
async def test_project_identity_by_slug_forwards_principal_and_project(
) -> None:
    project_id = ProjectID.generate()
    principal = MagicMock()
    request = MagicMock()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    expected = api.IdentityResponse(
        id='identity_' + 'a' * 32,
        slug='prod-aws',
        display_name='Production AWS',
        description='',
        timestamps=api.ResourceTimestamps(created_at=now, updated_at=now),
        credential_status=api.IdentityCredentialStatus(configured=False),
        config=api.S3IdentityConfig(
            authentication=api.S3AssumeRoleAuthentication(
                partition='aws',
                role_arn='arn:aws:iam::123456789012:role/production',
            ), ),
    )
    service = SimpleNamespace(available_for_project_by_slug=AsyncMock(
        return_value=expected))
    driver = MagicMock(spec=ConnectorApiDriver)
    driver._require_state_services.return_value = SimpleNamespace(
        identities=service)
    router = IdentityRouter(lambda: None, driver,
                            SimpleNamespace(require_customer_admin=AsyncMock()))

    with patch(
            'rest.public.routers.identity.get_user_authorization_from_request',
            return_value=principal):
        result = await router.get_available_by_slug(project_id, 'prod-aws',
                                                    request)

    assert result == expected
    service.available_for_project_by_slug.assert_awaited_once_with(
        project_id, 'prod-aws', principal)
    assert '/projects/{project_id}/identities/by-slug/{slug}' in {
        route.path
        for route in router.router.routes
    }


@pytest.mark.asyncio
async def test_project_identity_by_id_forwards_principal_and_project() -> None:
    project_id = ProjectID.generate()
    identity_id = IdentityID.generate()
    principal = MagicMock()
    request = MagicMock()
    request.state.user_authorization = principal
    expected = MagicMock(spec=api.IdentityResponse)
    service = SimpleNamespace(available_for_project_by_id=AsyncMock(
        return_value=expected))
    driver = MagicMock(spec=ConnectorApiDriver)
    driver._require_state_services.return_value = SimpleNamespace(
        identities=service)
    router = IdentityRouter(lambda: None, driver,
                            SimpleNamespace(require_customer_admin=AsyncMock()))

    result = await router.get_available_by_id(project_id, identity_id, request)

    assert result is expected
    service.available_for_project_by_id.assert_awaited_once_with(
        project_id, identity_id, principal)
    assert '/projects/{project_id}/identities/{identity_id}' in {
        route.path
        for route in router.router.routes
    }


@pytest.mark.asyncio
async def test_internal_project_identity_by_id_has_matching_route() -> None:
    project_id = ProjectID.generate()
    identity_id = IdentityID.generate()
    principal = MagicMock()
    request = MagicMock()
    request.state.user_authorization = principal
    expected = MagicMock(spec=api.IdentityResponse)
    service = SimpleNamespace(available_for_project_by_id=AsyncMock(
        return_value=expected))
    driver = MagicMock(spec=ConnectorApiDriver)
    driver._require_state_services.return_value = SimpleNamespace(
        identities=service)
    auth = SimpleNamespace(
        authenticate_request_header=lambda: None,
        require_customer_admin=lambda: None,
    )
    resource = IdentitiesResource(auth, driver)

    result = await resource.get_available_for_project_by_id(
        project_id, identity_id, request)

    assert result is expected
    service.available_for_project_by_id.assert_awaited_once_with(
        project_id, identity_id, principal)
    assert '/v0/projects/{project_id}/identities/{identity_id}' in {
        route.path
        for route in resource.project_router.routes
    }


# -- Administrator lifecycle on the SDK (/v1) surface ----------------------


def _admin_router(**service_methods):
    """Build an IdentityRouter with a stubbed identity service + admin auth."""
    service = SimpleNamespace(**service_methods)
    driver = MagicMock(spec=ConnectorApiDriver)
    driver._require_state_services.return_value = SimpleNamespace(
        identities=service)
    auth = SimpleNamespace(require_customer_admin=AsyncMock())
    router = IdentityRouter(lambda: None, driver, auth)
    return router, service, driver, auth


def _admin_paths(router):
    return {(route.path, tuple(sorted(route.methods)))
            for route in router.admin_router.routes}


@pytest.mark.asyncio
async def test_admin_create_forwards_args_and_principal() -> None:
    principal = MagicMock()
    request = MagicMock()
    args = MagicMock(spec=api.CreateIdentityArgs)
    expected = MagicMock(spec=api.IdentityResponse)
    router, service, _, _ = _admin_router(create=AsyncMock(
        return_value=expected))

    with patch(
            'rest.public.routers.identity.get_user_authorization_from_request',
            return_value=principal):
        result = await router.create(args, request)

    assert result is expected
    service.create.assert_awaited_once_with(args, principal)
    assert ('/identities', ('POST', )) in _admin_paths(router)


@pytest.mark.asyncio
async def test_admin_lifecycle_handlers_forward_to_service() -> None:
    principal = MagicMock()
    request = MagicMock()
    request.json = AsyncMock(return_value={
        'credentials': {
            'kind': 'databricks_oauth',
            'client_secret': 'replacement',
        }
    })
    identity_id = IdentityID.generate()
    resp = MagicMock(spec=api.IdentityResponse)
    groups = MagicMock(spec=api.IdentityGroupsResponse)
    update_args = MagicMock(spec=api.UpdateIdentityArgs)
    rotate_args = MagicMock(spec=api.RotateIdentityCredentialsArgs)
    groups_args = MagicMock(spec=api.ReplaceIdentityGroupsArgs)
    router, service, driver, _ = _admin_router(
        list_admin=AsyncMock(return_value=[resp]),
        get_admin=AsyncMock(return_value=resp),
        update_metadata=AsyncMock(return_value=resp),
        delete=AsyncMock(return_value=resp),
        rotate_credentials=AsyncMock(return_value=resp),
        replace_groups=AsyncMock(return_value=groups),
        list_groups=AsyncMock(return_value=groups),
    )
    driver.test_identity_credentials = AsyncMock(return_value=resp)

    with patch(
            'rest.public.routers.identity.get_user_authorization_from_request',
            return_value=principal):
        assert await router.list_admin(request) == [resp]
        assert await router.get_admin(identity_id, request) is resp
        assert await router.update(identity_id, update_args, request) is resp
        assert await router.delete(identity_id, request) is resp
        assert await router.rotate_credentials(identity_id, rotate_args,
                                               request) is resp
        assert await router.test_credentials(identity_id, request) is resp
        assert await router.replace_groups(identity_id, groups_args,
                                           request) is groups
        assert await router.list_groups(identity_id, request) is groups

    service.list_admin.assert_awaited_once_with(principal)
    service.get_admin.assert_awaited_once_with(identity_id, principal)
    service.update_metadata.assert_awaited_once_with(identity_id, update_args,
                                                     principal)
    service.delete.assert_awaited_once_with(identity_id, principal)
    service.rotate_credentials.assert_awaited_once_with(identity_id,
                                                        rotate_args,
                                                        principal,
                                                        preserve_s3_ca_certificate=False)
    driver.test_identity_credentials.assert_awaited_once_with(
        identity_id, principal)
    service.replace_groups.assert_awaited_once_with(identity_id, groups_args,
                                                    principal)
    service.list_groups.assert_awaited_once_with(identity_id, principal)


def test_admin_router_registers_full_lifecycle() -> None:
    router, _, _, _ = _admin_router()
    paths = _admin_paths(router)
    assert paths == {
        ('/identities', ('POST', )),
        ('/identities', ('GET', )),
        ('/identities/{identity_id}', ('GET', )),
        ('/identities/{identity_id}', ('PATCH', )),
        ('/identities/{identity_id}', ('DELETE', )),
        ('/identities/{identity_id}/credentials', ('PUT', )),
        ('/identities/{identity_id}/test', ('POST', )),
        ('/identities/{identity_id}/groups', ('PUT', )),
        ('/identities/{identity_id}/groups', ('GET', )),
    }


def test_admin_router_requires_customer_admin_gate() -> None:
    router, _, _, auth = _admin_router()
    admin_deps = {dep.dependency for dep in router.admin_router.dependencies}
    assert auth.require_customer_admin in admin_deps
    # The project-scoped discovery router must NOT require admin.
    read_deps = {dep.dependency for dep in router.router.dependencies}
    assert auth.require_customer_admin not in read_deps


@pytest.mark.asyncio
async def test_admin_routes_reject_non_admin_api_key() -> None:
    """End-to-end gate check: a non-admin principal is rejected with 403 and
    the identity service is never reached; an admin principal passes."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rest.config.authentication import AuthenticationSettings
    from rest.deps.authentication import Authentication

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    identity_id = IdentityID.generate()
    expected = api.IdentityResponse(
        id=str(identity_id),
        slug='prod-aws',
        display_name='Production AWS',
        description='',
        timestamps=api.ResourceTimestamps(created_at=now, updated_at=now),
        credential_status=api.IdentityCredentialStatus(configured=False),
        config=api.S3IdentityConfig(
            authentication=api.S3AssumeRoleAuthentication(
                partition='aws',
                role_arn='arn:aws:iam::123456789012:role/production',
            ), ),
    )
    service = SimpleNamespace(delete=AsyncMock(return_value=expected))
    driver = MagicMock(spec=ConnectorApiDriver)
    driver._require_state_services.return_value = SimpleNamespace(
        identities=service)
    auth = Authentication(
        settings=AuthenticationSettings(disable=False, is_unit_test=True,
                                        proxy=False), is_unit_test=True)
    router = IdentityRouter(lambda: None, driver, auth)

    def make_app(is_admin: bool) -> FastAPI:
        app = FastAPI()

        @app.middleware('http')
        async def attach_principal(request, call_next):
            request.state.user_authorization = SimpleNamespace(
                error=None, is_admin=is_admin)
            return await call_next(request)

        app.include_router(router.admin_router, prefix='/v1')
        return app

    non_admin = TestClient(make_app(is_admin=False))
    resp = non_admin.delete(f'/v1/identities/{identity_id}')
    assert resp.status_code == HTTPStatus.FORBIDDEN
    service.delete.assert_not_awaited()

    admin = TestClient(make_app(is_admin=True))
    resp = admin.delete(f'/v1/identities/{identity_id}')
    assert resp.status_code == HTTPStatus.OK
    service.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_v1_create_identity_then_connector_workflow() -> None:
    """The public /v1 HTTP surface the SDK calls (routers driven directly, not
    the kumoai_studio client): an admin creates an identity, then a project member
    creates a connector that references it. Members cannot create identities;
    admins can."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from kumoapi.json_serde import to_json_dict

    from rest.config.authentication import AuthenticationSettings
    from rest.deps.authentication import Authentication
    from rest.public.routers.connector import ConnectorRouter
    from statedb.config.common import ConnectorID

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    project_id = ProjectID.generate()
    identity_id = IdentityID.generate()

    identity_config = api.S3IdentityConfig(
        authentication=api.S3AssumeRoleAuthentication(
            partition='aws',
            role_arn='arn:aws:iam::123456789012:role/production',
        ))
    identity_response = api.IdentityResponse(
        id=str(identity_id),
        slug='prod-aws',
        display_name='Production AWS',
        description='',
        timestamps=api.ResourceTimestamps(created_at=now, updated_at=now),
        credential_status=api.IdentityCredentialStatus(configured=True),
        config=identity_config,
    )
    connector_config = api.S3ConnectorResourceConfig(
        target=api.S3RootedConnectorTarget(bucket='kumo-events',
                                           prefix='events', region='us-west-2'))
    connector_response = api.ConnectorResponse(
        id=str(ConnectorID.generate()),
        project_id=str(project_id),
        identity_id=str(identity_id),
        name='events',
        timestamps=api.ResourceTimestamps(created_at=now, updated_at=now),
        config=connector_config,
    )

    identity_service = SimpleNamespace(create=AsyncMock(
        return_value=identity_response))
    driver = MagicMock(spec=ConnectorApiDriver)
    driver._require_state_services.return_value = SimpleNamespace(
        identities=identity_service)
    driver.create_project_connector = AsyncMock(return_value=connector_response)

    auth = Authentication(
        settings=AuthenticationSettings(disable=False, is_unit_test=True,
                                        proxy=False), is_unit_test=True)
    identity_router = IdentityRouter(lambda: None, driver, auth)
    connector_router = ConnectorRouter(lambda: None, driver)

    def make_app(is_admin: bool) -> FastAPI:
        app = FastAPI()

        @app.middleware('http')
        async def attach_principal(request, call_next):
            request.state.user_authorization = SimpleNamespace(
                error=None, is_admin=is_admin)
            return await call_next(request)

        app.include_router(identity_router.admin_router, prefix='/v1')
        app.include_router(connector_router.router, prefix='/v1')
        return app

    identity_body = to_json_dict(
        api.CreateIdentityArgs(display_name='Production AWS',
                               config=identity_config))
    connector_body = to_json_dict(
        api.CreateConnectorArgs(name='events', config=connector_config,
                                identity_id=identity_id))
    connectors_url = f'/v1/projects/{project_id}/connectors'

    # Admin: full workflow succeeds end-to-end.
    admin = TestClient(make_app(is_admin=True))
    created = admin.post('/v1/identities', json=identity_body)
    assert created.status_code == HTTPStatus.OK
    assert created.json()['id'] == str(identity_id)
    made = admin.post(connectors_url, json=connector_body)
    assert made.status_code == HTTPStatus.OK
    assert made.json()['identity_id'] == str(identity_id)

    # Member: cannot create identities, but can create connectors that
    # reference an admin-provisioned one.
    member = TestClient(make_app(is_admin=False))
    assert member.post('/v1/identities',
                       json=identity_body).status_code == HTTPStatus.FORBIDDEN
    assert member.post(connectors_url,
                       json=connector_body).status_code == HTTPStatus.OK
