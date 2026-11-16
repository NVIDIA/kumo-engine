# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from http import HTTPStatus
from unittest.mock import MagicMock

import databricks.sql.exc as DatabricksErrors
import kumoapi.data_source as api
import pytest
from fastapi.testclient import TestClient
from kumoapi.data_source import DEFAULT_PROJECT_ID
from pydantic import SecretStr

from kumo.connector.databricks_connector import DatabricksConnector
from rest.deps.authentication import UserAuthorization
from statedb.config.common import ProjectID


def _cursor(*, rows=None, execute_error=None):
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    if execute_error is not None:
        cursor.execute.side_effect = execute_error
    else:
        result = cursor.execute.return_value
        result.fetchall.return_value = rows or []
        result.fetchone.return_value = (1, )
    return cursor


def _server_error(code: str, detail: str = 'provider secret detail'):
    return DatabricksErrors.ServerOperationError(f'[{code}] {detail}')


async def _create_bound_identity(state_services, monkeypatch) -> str:
    # The public test app is auth-disabled even though the session deploy
    # config intentionally omits an auth block. Keep the worker-side current
    # authority resolver on the same trusted mode for this route test.
    monkeypatch.setenv('DISABLE_AUTH', 'true')
    admin = UserAuthorization.disabled()
    project = await state_services.projects.get_project(
        ProjectID(DEFAULT_PROJECT_ID), admin)
    monkeypatch.setattr(state_services.identities, '_identity_validator',
                        lambda *_args: None)
    identity = await state_services.identities.create(
        api.CreateIdentityArgs(
            display_name='Samples OAuth identity',
            config=api.DatabricksIdentityConfig(
                host='https://workspace.cloud.databricks.com',
                client_id='samples-client-id',
            ),
            credentials=api.DatabricksOAuthCredentials(
                client_secret=SecretStr('samples-client-secret')),
        ),
        admin,
    )
    await state_services.identities.replace_groups(
        identity.id,
        api.ReplaceIdentityGroupsArgs(group_ids=[str(project.group_id)]),
        admin,
    )
    return str(identity.id)


def _create_request(identity_id: str, catalog: str = 'samples') -> dict:
    return {
        'identity_id': identity_id,
        'name': f'{catalog} tpch',
        'config': {
            'type': 'databricks',
            'warehouse_id': 'warehouse-id',
            'catalog': catalog,
            'schema_name': 'tpch',
        },
    }


def test_samples_catalog_create_uses_exact_read_fallback(
    public_api_client: TestClient,
    state_services,
    monkeypatch,
) -> None:
    identity_id = asyncio.run(
        _create_bound_identity(state_services, monkeypatch))
    grant_cursor = _cursor(
        execute_error=_server_error('SAMPLE_TABLE_PERMISSIONS'))
    schema_cursor = _cursor(rows=[('information_schema', ), ('tpch', )])
    table_cursor = _cursor(rows=[('tpch', 'region', False)])
    read_cursor = _cursor()
    context = MagicMock()
    context.cursor.side_effect = [
        grant_cursor,
        schema_cursor,
        table_cursor,
        read_cursor,
    ]
    open_provider = MagicMock(return_value=context)
    close_provider = MagicMock()
    monkeypatch.setattr(DatabricksConnector, '_open', open_provider)
    monkeypatch.setattr(DatabricksConnector, '_close', close_provider)

    response = public_api_client.post(
        f'/v1/projects/{DEFAULT_PROJECT_ID}/connectors',
        json=_create_request(identity_id),
    )

    assert response.status_code == HTTPStatus.OK, response.text
    grant_cursor.execute.assert_called_once_with(
        'show grants `samples-client-id` on SCHEMA `samples`.`tpch`')
    schema_cursor.execute.assert_called_once_with('SHOW SCHEMAS IN `samples`')
    table_cursor.execute.assert_called_once_with(
        'SHOW TABLES IN `samples`.`tpch`')
    read_cursor.execute.assert_called_once_with(
        'SELECT 1 FROM `samples`.`tpch`.`region` LIMIT 1')
    open_provider.assert_called_once()
    close_provider.assert_called_once_with(context)


@pytest.mark.parametrize(
    ('catalog', 'grant_error', 'schema_rows'),
    [
        ('main', _server_error('SAMPLE_TABLE_PERMISSIONS'), None),
        ('samples', _server_error('PERMISSION_DENIED'), None),
        (
            'samples',
            DatabricksErrors.ServerOperationError(
                'arbitrary SAMPLE_TABLE_PERMISSIONS provider secret'),
            None,
        ),
        ('samples', _server_error('SAMPLE_TABLE_PERMISSIONS'), [('private', )]),
    ],
    ids=['customer-catalog', 'other-code', 'unstructured-code',
         'schema-not-visible'],
)
def test_samples_catalog_create_fails_closed_for_every_other_case(
    public_api_client: TestClient,
    state_services,
    monkeypatch,
    catalog,
    grant_error,
    schema_rows,
) -> None:
    identity_id = asyncio.run(
        _create_bound_identity(state_services, monkeypatch))
    grant_cursor = _cursor(execute_error=grant_error)
    cursors = [grant_cursor]
    if schema_rows is not None:
        cursors.append(_cursor(rows=schema_rows))
    context = MagicMock()
    context.cursor.side_effect = cursors
    open_provider = MagicMock(return_value=context)
    close_provider = MagicMock()
    monkeypatch.setattr(DatabricksConnector, '_open', open_provider)
    monkeypatch.setattr(DatabricksConnector, '_close', close_provider)

    response = public_api_client.post(
        f'/v1/projects/{DEFAULT_PROJECT_ID}/connectors',
        json=_create_request(identity_id, catalog),
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert 'provider secret' not in response.text
    open_provider.assert_called_once()
    close_provider.assert_called_once_with(context)
