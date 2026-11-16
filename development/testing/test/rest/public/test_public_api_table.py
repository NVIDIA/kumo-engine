# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from kumoapi.data_source import DEFAULT_PROJECT_ID

from rest.driver.table_driver import TableDriver


def _table_definition(source_path: str, connector_id: str) -> dict:
    return {
        'cols': [{
            'name': 'event_id',
            'stype': 'ID',
            'dtype': 'int64',
        }],
        'source_table': {
            'connector_id': connector_id,
            'source': {
                'mode': 'relative',
                'path': source_path,
            },
            'file_type': 'PARQUET',
            'data_source_type': 'S3',
        },
        'pkey': 'event_id',
        'time_col': None,
        'end_time_col': None,
    }


def _expect_success(response) -> dict:
    assert response.status_code == HTTPStatus.OK, response.text
    return response.json()


def test_public_table_create_and_overwrite_through_http(
    public_api_client: TestClient,
    monkeypatch,
) -> None:
    """Exercise the production public route and state services without cloud.

    Source validation is the only isolated boundary; table persistence,
    project authorization, create-or-replace behavior, and serialization all
    run through the production HTTP application.
    """
    validate_source = AsyncMock()
    monkeypatch.setattr(
        TableDriver,
        '_validate_new_table_for_state_service',
        validate_source,
    )
    upload_connector = _expect_success(
        public_api_client.get(
            f'/v1/projects/{DEFAULT_PROJECT_ID}/connectors/uploads/connector'))
    connector_id = upload_connector['connector_id']
    params = {
        'project_id': DEFAULT_PROJECT_ID,
        'name_alias': 'sdk_smoke_table',
        'force_rename': 'true',
    }

    first_definition = _table_definition('first/events.parquet', connector_id)
    first = _expect_success(
        public_api_client.post('/v1/tables', params=params,
                               json=first_definition))

    second_definition = _table_definition('second/events.parquet', connector_id)
    second = _expect_success(
        public_api_client.post('/v1/tables', params=params,
                               json=second_definition))
    assert second['id'] != first['id']

    loaded = _expect_success(
        public_api_client.get(
            '/v1/tables/sdk_smoke_table',
            params={'project_id': DEFAULT_PROJECT_ID},
        ))
    assert loaded['name_alias'] == 'sdk_smoke_table'
    assert loaded['table']['source_table']['source'] == {
        'mode': 'relative',
        'path': 'second/events.parquet',
    }

    # A nameless table is filtered out of the list read above and no delete
    # route accepts its identifier, so the create route refuses it.
    refused = public_api_client.post(
        '/v1/tables',
        params={'project_id': DEFAULT_PROJECT_ID},
        json=_table_definition('unnamed/events.parquet', connector_id),
    )
    assert refused.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, refused.text
    assert 'name_alias' in refused.text

    # The refusal precedes any work: source validation is never reached.
    assert validate_source.await_count == 2
