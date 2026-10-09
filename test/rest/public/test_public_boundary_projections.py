# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, fields
from datetime import datetime, timezone
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock

import kumoapi.data_snapshot as snapshot_api
import kumoapi.pquery as pquery_api
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kumoapi.common import ValidationResponse
from kumoapi.graph import GraphDefinition
from kumoapi.source_table import UnavailableSourceTable
from kumoapi.table import TableDefinition as ApiTableDefinition

from rest.config.predictive_query_v2 import PQueryResource
from rest.config.table_v2 import TableDefinition as InternalTableDefinition
from rest.config.table_v2 import TableSnapshotResource
from rest.deps.authentication import Authentication
from rest.public.routers.data_snapshot import _public_table_snapshot
from rest.public.routers.pquery import (
    PQueryRouter,
    _internal_pquery_from_public,
    _public_pquery_from_internal,
)
from statedb.config.common import ProjectID
from statedb.config.timestamp import ResourceTimestamps

_NOW = datetime(2025, 1, 2, tzinfo=timezone.utc)
_PROJECT_ID = ProjectID('default')


def _public_pquery() -> pquery_api.PQueryResource:
    return pquery_api.PQueryResource(
        query_string='PREDICT events.label FOR EACH events.id',
        graph=GraphDefinition(tables={}, col_groups=[]),
        name='events_query',
        desc='public description',
    )


def test_pquery_request_and_response_use_exact_public_fields() -> None:
    public = _public_pquery()
    enriched = _internal_pquery_from_public(public)

    assert asdict(enriched) == {
        'query_string': public.query_string,
        'graph': asdict(public.graph),
        'name': public.name,
        'desc': public.desc,
        'graph_display_name': None,
        'advanced_options_yaml': None,
        'runtime_options': None,
    }
    internal = PQueryResource(
        query_string=public.query_string,
        graph=public.graph,
        name=public.name,
        desc=public.desc,
        graph_display_name='internal graph name',
        advanced_options_yaml='internal yaml',
        runtime_options={
            'list_distinct_target_is_array': True,
            'lp_bp_layout': 'internal-layout',
        },
    )

    projected = _public_pquery_from_internal(internal)

    assert isinstance(projected, pquery_api.PQueryResource)
    assert type(projected.graph) is GraphDefinition
    assert {field.name
            for field in fields(projected)
            } == {'query_string', 'graph', 'name', 'desc'}
    payload = asdict(projected)
    assert 'internal graph name' not in repr(payload)
    assert 'internal yaml' not in repr(payload)
    assert 'internal-layout' not in repr(payload)


@pytest.fixture
def pquery_boundary_client():
    driver = AsyncMock()
    auth = MagicMock(spec=Authentication)
    router = PQueryRouter(lambda: None, driver, auth)
    app = FastAPI()
    app.include_router(router.router, prefix='/v1')
    with TestClient(app) as client:
        yield client, driver, app


@pytest.mark.parametrize(
    'internal_field',
    [
        'runtime_options',
        'advanced_options_yaml',
        'graph_display_name',
        'list_distinct_target_is_array',
        'lp_bp_layout',
    ],
)
def test_pquery_create_and_validate_reject_internal_fields_with_422(
    pquery_boundary_client,
    internal_field: str,
) -> None:
    client, driver, _ = pquery_boundary_client
    payload = asdict(_public_pquery())
    payload[internal_field] = {}
    for path in [
            '/v1/predictive_queries',
            '/v1/predictive_queries/validate',
    ]:
        response = client.post(
            path,
            params={'project_id': str(_PROJECT_ID)},
            json=payload,
        )
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    driver.create_predictive_query_v2.assert_not_awaited()
    driver.validate_predictive_query_v2.assert_not_awaited()


def test_pquery_openapi_schema_is_exact_and_strict(
    pquery_boundary_client, ) -> None:
    _, _, app = pquery_boundary_client
    schema = app.openapi()['components']['schemas']['PQueryResource']
    assert schema['additionalProperties'] is False
    assert set(
        schema['properties']) == {'query_string', 'graph', 'name', 'desc'}


@pytest.mark.asyncio
async def test_pquery_validate_get_and_list_use_boundary_projection() -> None:
    router = PQueryRouter.__new__(PQueryRouter)
    router._driver = AsyncMock()
    public = _public_pquery()
    internal = PQueryResource(
        query_string=public.query_string,
        graph=public.graph,
        name=public.name,
        desc=public.desc,
        advanced_options_yaml='must-not-leak',
        runtime_options={'list_distinct_target_is_array': True},
    )
    router._driver.validate_predictive_query_v2.return_value = (
        ValidationResponse())
    router._driver.get_predictive_query_v2.return_value = internal
    router._driver.list_predictive_queries_v2.return_value = [internal]

    await router.validate_pquery(public, str(_PROJECT_ID))
    validation_request = (
        router._driver.validate_predictive_query_v2.await_args.kwargs['args'])
    assert validation_request.pquery_resource.runtime_options is None
    assert validation_request.pquery_resource.advanced_options_yaml is None

    got = await router.get_pquery(public.name, str(_PROJECT_ID))
    listed = await router.list_pqueries(str(_PROJECT_ID))
    assert asdict(got) == asdict(public)
    assert [asdict(resource) for resource in listed] == [asdict(public)]
    assert 'must-not-leak' not in repr((got, listed))


def test_table_snapshot_projection_is_exact_and_drops_internal_metadata(
) -> None:
    table_definition = InternalTableDefinition(
        cols=[],
        source_table=UnavailableSourceTable(table='events'),
    )
    internal = TableSnapshotResource(
        id=snapshot_api.TableSnapshotID('table@version'),
        table_definition=table_definition,
        stats=None,
        column_stats=None,
        stages={},
        timestamps=ResourceTimestamps(created_at=_NOW, updated_at=_NOW),
        connector=None,
        name_alias='internal-name',
        time_range=None,
    )

    public = _public_table_snapshot(internal)

    assert isinstance(public, snapshot_api.TableSnapshotResource)
    assert type(public.table_definition) is ApiTableDefinition
    assert type(public.table_definition.source_table) is UnavailableSourceTable
    assert set(asdict(public)) == {
        'id', 'table_definition', 'stats', 'column_stats', 'stages'
    }
    assert 'internal-name' not in repr(asdict(public))


def test_removed_public_utility_routes_are_absent(
    public_api_client: TestClient, ) -> None:
    assert public_api_client.get('/utils/persist_dir').status_code == (
        HTTPStatus.NOT_FOUND)
    response = public_api_client.post(
        '/utils/refresh_connector_snapshot_data',
        json={
            'snapshot_root_dir': 's3://source',
            'connector_root_dir': '/destination',
            'slack_webhook_url': 'https://attacker.invalid',
        },
    )
    assert response.status_code == HTTPStatus.NOT_FOUND
