# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import kumoapi.source_table as source_table_api
import kumoapi.table as table_api
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from kumo_core.types.schema import FPID
from kumoapi.common import ValidationError
from kumoapi.json_serde import to_json_dict
from kumoapi.typing import Dtype, Stype

from rest.config.graph_v2 import ListResponse
from rest.config.status import EmptyResponse
from rest.config.table_v2 import FileType
from rest.public.routers.table import TableRouter as PublicTableRouter
from rest.routers.tables_v2 import TableRouter

_PROJECT_ID = "project_" + "1" * 32
_CONNECTOR_ID = "connector_" + "1" * 32


@pytest.fixture(autouse=True, scope="session")
def setup_rest_test_session():
    yield


@pytest.fixture(autouse=True, scope="session")
def temporal_workers_ready():
    yield True


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def table_router():
    auth = SimpleNamespace(authenticate_request_header=lambda: None)
    table_driver = AsyncMock()
    return TableRouter(auth, table_driver)


def _table_definition() -> table_api.TableDefinition:
    return table_api.TableDefinition(
        cols=[
            table_api.Column(
                name="id",
                stype=Stype.ID,
                dtype=Dtype.string,
                timestamp_format=None,
            )
        ],
        source_table=source_table_api.S3SourceTable(
            connector_id=_CONNECTOR_ID,
            source={
                'mode': 'relative',
                'path': "orders"
            },
            file_type=FileType.PARQUET,  # pyright: ignore
            data_source_type=source_table_api.DataSourceType.S3,
        ),
        pkey="id",
        time_col=None,
        end_time_col=None,
    )


def _metadata_request() -> table_api.TableMetadataRequest:
    return table_api.TableMetadataRequest(
        cols=[],
        source_table=source_table_api.S3SourceTable(
            connector_id=_CONNECTOR_ID,
            source={
                'mode': 'relative',
                'path': "orders"
            },
            file_type=FileType.PARQUET,  # pyright: ignore
            data_source_type=source_table_api.DataSourceType.S3,
        ),
    )


def _metadata_response(
        request: table_api.TableMetadataRequest
) -> table_api.TableMetadataResponse:
    return table_api.TableMetadataResponse(
        cols=[
            table_api.Column(
                name="description",
                stype=Stype.text,
                dtype=Dtype.string,
                timestamp_format=None,
            )
        ],
        source_table=request.source_table,
        pkey=None,
        time_col=None,
        end_time_col=None,
    )


def _sql_resolve_request() -> table_api.SqlTableResolveRequest:
    return table_api.SqlTableResolveRequest(
        source_table=source_table_api.CustomSQLSourceTable(
            connector_id=_CONNECTOR_ID,
            sql="SELECT id FROM orders",
            table="pending",
        ),
        column_hints=None,
        pkey="id",
    )


@pytest.mark.asyncio
async def test_infer_metadata_dispatches_the_sdk_inference(
        table_router: TableRouter):
    """The wizard route delegates to the same stats-based
    ``TableDriver.infer_table_metadata`` the public SDK endpoint dispatches,
    so UI- and SDK-authored tables infer identical default stypes for the
    same source (kumo-ai/kumo#30131)."""
    request = _metadata_request()
    expected = _metadata_response(request)
    table_router._table_driver.infer_table_metadata.return_value = expected

    actual = await table_router.infer_metadata(request, project_id=_PROJECT_ID)

    table_router._table_driver.infer_table_metadata.assert_called_once_with(
        request, project_id=_PROJECT_ID)
    assert actual is expected

    # Parity at the seam: the public SDK route produces the identical driver
    # dispatch (and therefore the identical inference) for the same request.
    public_router = PublicTableRouter(
        api_key_auth=lambda: None,
        table_driver=table_router._table_driver,
        auth=SimpleNamespace(),
    )
    sdk_response = await public_router.infer_metadata(request,
                                                      project_id=_PROJECT_ID)

    assert sdk_response is expected
    driver_mock = table_router._table_driver.infer_table_metadata
    assert driver_mock.call_count == 2
    assert driver_mock.call_args_list[0] == driver_mock.call_args_list[1]


@pytest.mark.asyncio
async def test_infer_metadata_route_over_asgi(table_router: TableRouter):
    """The wizard-facing route parses a real request and serializes the
    stats-based response over the internal ASGI app."""
    request = _metadata_request()
    table_router._table_driver.infer_table_metadata.return_value = (
        _metadata_response(request))

    app = FastAPI()
    app.include_router(table_router.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport,
                           base_url='http://test') as client:
        response = await client.post(
            '/v1/tables/infer_metadata',
            params={'project_id': _PROJECT_ID},
            json=to_json_dict(_metadata_request()),
        )

    assert response.status_code == 200
    table_router._table_driver.infer_table_metadata.assert_awaited_once_with(
        request, project_id=_PROJECT_ID)
    assert response.json()['cols'][0]['stype'] == 'text'


@pytest.mark.asyncio
async def test_public_resolve_sql_route_dispatches_over_asgi(
        table_router: TableRouter):
    request = _sql_resolve_request()
    expected = table_api.SqlTableResolveResponse(
        table_definition=None,
        preview_rows=[],
        errors=[
            ValidationError(title="Invalid Custom SQL table",
                            message="fixture error")
        ],
    )
    table_router._table_driver.resolve_sql.return_value = expected
    public_router = PublicTableRouter(
        api_key_auth=lambda: None,
        table_driver=table_router._table_driver,
        auth=SimpleNamespace(),
    )

    app = FastAPI()
    app.include_router(public_router.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport,
                           base_url='http://test') as client:
        response = await client.post(
            '/tables/resolve_sql',
            params={'project_id': _PROJECT_ID},
            json=to_json_dict(request),
        )

    assert response.status_code == 200
    table_router._table_driver.resolve_sql.assert_awaited_once_with(
        request, project_id=_PROJECT_ID)
    assert response.json()['errors'][0]['message'] == "fixture error"


@pytest.mark.asyncio
async def test_list_table_names_project_scoped(table_router: TableRouter):
    expected = [
        ListResponse(name="orders", id="NA.NA-abc", connector_id=None,
                     source_table_name="orders", timestamps=None)
    ]
    table_router._table_driver.list_project_table_names.return_value = expected

    actual = await table_router.list_table_names(project_id=_PROJECT_ID)

    table_router._table_driver.list_project_table_names.assert_called_once_with(
        _PROJECT_ID)
    assert actual == expected


@pytest.mark.asyncio
async def test_get_col_stats_project_scoped(table_router: TableRouter):
    fpid = FPID("a" * 32)
    resolved_id = f"NA.NA-{fpid}"
    resolved = SimpleNamespace(id=fpid)
    expected = object()
    table_router._table_driver.resolve_project_table.return_value = resolved
    table_router._table_driver.get_col_stats_and_sample_rows.return_value = (
        expected)

    actual = await table_router.get_col_stats("orders", project_id=_PROJECT_ID)

    table_router._table_driver.resolve_project_table.assert_called_once_with(
        _PROJECT_ID, "orders")
    table_router._table_driver.get_col_stats_and_sample_rows.assert_called_once_with(
        resolved_id, project_id=_PROJECT_ID)
    assert actual is expected


@pytest.mark.asyncio
async def test_refresh_table_project_scoped(table_router: TableRouter):
    fpid = FPID("a" * 32)
    resolved_id = f"NA.NA-{fpid}"
    resolved = SimpleNamespace(id=fpid)
    table_router._table_driver.resolve_project_table.return_value = resolved

    actual = await table_router.refresh_table_v2("orders",
                                                 project_id=_PROJECT_ID)

    table_router._table_driver.resolve_project_table.assert_called_once_with(
        _PROJECT_ID, "orders")
    table_router._table_driver.refresh_table.assert_called_once_with(
        resolved_id, project_id=_PROJECT_ID)
    assert isinstance(actual, EmptyResponse)


@pytest.mark.asyncio
async def test_get_vals_project_scoped(table_router: TableRouter):
    # FPID must be 32 hex chars (matches production-generated fpids).
    resolved_id = f"NA.NA-{'a' * 32}"
    resolved = SimpleNamespace(id=FPID("a" * 32))
    expected = object()
    table_router._table_driver.resolve_project_table.return_value = resolved
    table_router._table_driver.get_table_validation.return_value = expected

    actual = await table_router.get_vals(
        "orders",
        project_id=_PROJECT_ID,
        data_version="v1",
    )

    table_router._table_driver.resolve_project_table.assert_called_once_with(
        _PROJECT_ID, "orders")
    table_router._table_driver.get_table_validation.assert_called_once_with(
        resolved_id, _PROJECT_ID, "v1")
    assert actual is expected


@pytest.mark.asyncio
async def test_get_table_v2_returns_canonical_project_resource(
    table_router: TableRouter, ):
    valid_fpid = "a" * 32  # FPID must be 32 hex chars
    resolved = SimpleNamespace(
        id=FPID(valid_fpid),
        project_id=_PROJECT_ID,
        alias="orders",
        created_by_name="Alice Smith",
        timestamps=SimpleNamespace(),
    )
    response = SimpleNamespace(
        id=valid_fpid,
        table=None,
        name_alias="orders",
        timestamps=resolved.timestamps,
        stats=None,
        time_range=None,
        connector=None,
    )
    table_router._table_driver.resolve_project_table.return_value = resolved
    table_router._table_driver.get_table.return_value = response

    actual = await table_router.get_table_v2("orders", project_id=_PROJECT_ID)

    table_router._table_driver.resolve_project_table.assert_called_once_with(
        _PROJECT_ID, "orders")
    table_router._table_driver.get_table.assert_called_once_with(
        FPID(valid_fpid), _PROJECT_ID)
    assert actual.name_alias == "orders"
    assert actual.timestamps == resolved.timestamps


@pytest.mark.asyncio
async def test_validate_name_project_scoped(table_router: TableRouter):
    expected = SimpleNamespace(is_valid=True, error_msg=None)
    table_router._table_driver.validate_project_table_name.return_value = (
        expected)

    actual = await table_router.validate_v2_table_name(
        "orders",
        project_id=_PROJECT_ID,
    )

    table_router._table_driver.validate_project_table_name.assert_called_once_with(
        project_id=_PROJECT_ID, name="orders")
    assert actual is expected


@pytest.mark.asyncio
async def test_create_table_project_scoped(table_router: TableRouter):
    """Happy path passes the one project-local alias to the driver."""
    table_router._table_driver.create_full_table.return_value = "NA.NA-1234abcd"

    response = await table_router.create_table_v2(
        table=_table_definition(),
        project_id=_PROJECT_ID,
        name_alias="orders",
    )

    table_router._table_driver.create_full_table.assert_called_once_with(
        ANY,
        alias="orders",
        force_rename=False,
        project_id=_PROJECT_ID,
    )
    assert response.id == "NA.NA-1234abcd"


@pytest.mark.asyncio
@pytest.mark.parametrize('removed_field',
                         ('name_alias_in_project', 'display_name_in_project'))
async def test_create_table_rejects_removed_query_fields_over_asgi(
    table_router: TableRouter,
    removed_field: str,
) -> None:
    app = FastAPI()
    app.include_router(table_router.router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport,
                           base_url='http://test') as client:
        response = await client.post(
            '/v1/tables/',
            params={
                'project_id': _PROJECT_ID,
                removed_field: 'ignored-before-strict-validation',
            },
            json=to_json_dict(_table_definition()),
        )

    assert response.status_code == 422
    assert removed_field in response.text
    table_router._table_driver.create_full_table.assert_not_awaited()
