# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import kumoapi.data_source as api
import kumoapi.source_table as source_table_api
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import TimeoutError as TemporalTimeoutError
from temporalio.exceptions import TimeoutType

from rest.deps.authentication import Authentication
from rest.driver.connector_api_driver import (
    ConnectorApiDriver,
    get_connector_sample_rows,
    get_connector_source_columns,
)
from rest.public.routers.connector import ConnectorRouter
from rest.public.routers.source_table import SourceTableRouter
from rest.routers.connectors import ConnectorResource
from statedb.config.common import ConnectorID, IdentityID, ProjectID
from statedb.utils.errors import (
    ConnectorDoesNotExistError,
    ConnectorNameConflictError,
    ConnectorSourceTableNotFoundError,
    ConnectorValidationError,
    ExternalMembershipUnavailableError,
    IdentityDoesNotExistError,
    IdentityResolutionUnavailableError,
    MultipleIdentitiesValidateTargetError,
    NoIdentityValidatesTargetError,
)


@pytest.mark.parametrize('resource_type', [ConnectorResource, ConnectorRouter])
@pytest.mark.asyncio
async def test_connector_project_membership_outage_is_generic_503(
        resource_type) -> None:
    service = SimpleNamespace(list_for_project=AsyncMock(
        side_effect=ExternalMembershipUnavailableError(
            'External group membership is unavailable; please retry')))
    driver = object.__new__(ConnectorApiDriver)
    driver._state_services = SimpleNamespace(connectors=service)
    resource = object.__new__(resource_type)
    resource._driver = driver
    resource._principal = lambda _request: 'unresolved-principal'

    with pytest.raises(HTTPException) as caught:
        await resource.list_connectors(ProjectID.generate(), object())

    assert caught.value.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert caught.value.detail == (
        'External group membership is unavailable; please retry')
    assert 'provider' not in str(caught.value.detail).lower()


@pytest.fixture(autouse=True, scope='session')
def setup_rest_test_session():
    yield


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def resource():
    value = object.__new__(ConnectorResource)
    value._driver = SimpleNamespace(
        get_project_source_table_names=AsyncMock(return_value=['orders']),
        get_project_source_columns=AsyncMock(return_value=SimpleNamespace(
            source_cols=[])),
        get_project_sample_rows=AsyncMock(return_value=SimpleNamespace(
            columns=[],
            sample_df_serialized='{"data": []}',
        )),
    )
    value._principal = lambda _request: 'principal'
    return value


@pytest.fixture
def source_table_route():
    def build(config, file_type=source_table_api.FileType.PARQUET):
        project_id = ProjectID.generate()
        connector_id = ConnectorID.generate()
        service = SimpleNamespace(source_table_connector_for_project=AsyncMock(
            return_value=SimpleNamespace(config=config)))
        driver = object.__new__(ConnectorApiDriver)
        driver._state_services = SimpleNamespace(connectors=service)
        driver.get_project_source_table_file_type = AsyncMock(
            return_value=(file_type, None))
        route = SourceTableRouter(lambda: None, driver,
                                  MagicMock(spec=Authentication))
        route._principal = lambda _request: 'principal'
        app = FastAPI()
        app.include_router(route.router, prefix='/v1')
        return SimpleNamespace(
            app=app,
            connector_id=connector_id,
            driver=driver,
            project_id=project_id,
            service=service,
        )

    return build


@pytest.mark.asyncio
async def test_discovery_uses_exact_project_connector_and_principal(
        resource) -> None:
    project_id = ProjectID.generate()
    connector_id = ConnectorID.generate()

    source_tables = await resource.list_source_tables(project_id, connector_id,
                                                      object())
    nested_name = 'archive/2026/07/orders.parquet'
    columns = await resource.get_columns(project_id, connector_id, object(),
                                         table_name=nested_name)
    sample = await resource.get_sample(
        project_id,
        connector_id,
        object(),
        table_name=nested_name,
        sample_rows=7,
    )

    assert source_tables.table_names == ['orders']
    assert columns.table_name == nested_name
    assert sample.sample_rows == '{"data": []}'
    resource._driver.get_project_source_table_names.assert_awaited_once_with(
        project_id, connector_id, 'principal')
    resource._driver.get_project_source_columns.assert_awaited_once_with(
        project_id,
        connector_id,
        'principal',
        nested_name,
        skip_cache=False,
        query=None,
    )
    resource._driver.get_project_sample_rows.assert_awaited_once_with(
        project_id,
        connector_id,
        'principal',
        nested_name,
        num_rows=7,
        serialize_to_json=True,
        require_stats=False,
        query=None,
    )


@pytest.mark.parametrize(
    ('data_plane_storage_root', 'source_table_type', 'relative_source_type'),
    [
        ('/local/storage', source_table_api.S3SourceTable,
         source_table_api.S3RelativeSource),
        ('s3://data/root', source_table_api.S3SourceTable,
         source_table_api.S3RelativeSource),
        ('gs://data/root', source_table_api.GCSSourceTable,
         source_table_api.GCSRelativeSource),
        (
            'abfss://data@account.dfs.core.windows.net/root',
            source_table_api.ADLSSourceTable,
            source_table_api.ADLSRelativeSource,
        ),
    ],
)
@pytest.mark.asyncio
async def test_public_source_table_config_supports_project_upload_connector(
    monkeypatch,
    data_plane_storage_root,
    source_table_type,
    relative_source_type,
) -> None:
    project_id = ProjectID.generate()
    connector_id = ConnectorID.generate()
    service = SimpleNamespace(
        source_table_connector_for_project=AsyncMock(return_value=None), )
    uploaded_files = SimpleNamespace(get_completed=AsyncMock(
        return_value=SimpleNamespace(file_type=api.UploadFileType.CSV)))
    driver = object.__new__(ConnectorApiDriver)
    driver._state_services = SimpleNamespace(
        connectors=service,
        uploaded_files=uploaded_files,
    )
    resource = object.__new__(SourceTableRouter)
    resource._driver = driver
    resource._principal = lambda _request: 'principal'
    monkeypatch.setattr(
        'rest.driver.connector_api_driver.deploy_env',
        lambda: SimpleNamespace(data_plane_storage_root=data_plane_storage_root
                                ),
    )

    response = await resource.get_source_table_config(
        project_id,
        connector_id,
        object(),
        'orders',
    )

    assert response.source_table == source_table_type(
        connector_id=str(connector_id),
        source=relative_source_type(path='orders'),
        file_type=source_table_api.FileType.CSV,
    )
    service.source_table_connector_for_project.assert_awaited_once_with(
        project_id,
        connector_id,
        'principal',
    )
    uploaded_files.get_completed.assert_awaited_once_with(
        project_id,
        connector_id,
        'orders',
    )


@pytest.mark.parametrize(
    ('config', 'data_source_type', 'table_name', 'file_type'),
    [
        (
            api.S3ConnectorResourceConfig(target=api.S3RootedConnectorTarget(
                bucket='company-data', prefix='project-root')),
            'S3',
            'nested/2026/07/dataset.csv/',
            source_table_api.FileType.CSV,
        ),
        (
            api.GCSConnectorResourceConfig(bucket='company-data',
                                           prefix='project-root'),
            'GCS',
            'nested/2026/07/dataset.parquet/',
            source_table_api.FileType.PARQUET,
        ),
        (
            api.ADLSConnectorResourceConfig(
                storage_account='companylake',
                filesystem='curated',
                prefix='project-root',
            ),
            'ADLS',
            'nested/2026/07/dataset.csv/',
            source_table_api.FileType.CSV,
        ),
    ],
)
def test_public_source_table_config_preserves_rooted_directory_intent(
        source_table_route, config, data_source_type, table_name,
        file_type) -> None:
    route = source_table_route(config, file_type)

    with TestClient(route.app) as client:
        response = client.get(
            f'/v1/projects/{route.project_id}/connectors/{route.connector_id}/'
            'source-table-config', params={'table_name': table_name})

    assert response.status_code == 200
    assert response.json()['source_table']['source'] == {
        'mode': 'relative',
        'path': table_name,
    }
    assert response.json()['source_table']['data_source_type'] == (
        data_source_type)
    assert response.json()['source_table']['file_type'] == file_type.value
    route.service.source_table_connector_for_project.assert_awaited_once_with(
        route.project_id, route.connector_id, 'principal')
    route.driver.get_project_source_table_file_type.assert_awaited_once_with(
        route.project_id,
        route.connector_id,
        'principal',
        table_name,
    )


@pytest.mark.parametrize(
    'table_name',
    [
        '/absolute/table.parquet',
        'nested//table.parquet',
        'nested/./table.parquet',
        'nested/../table.parquet',
        'nested\\table.parquet',
    ],
)
def test_public_source_table_rejects_non_relative_or_traversing_path(
        table_name) -> None:
    with pytest.raises(ValueError):
        SourceTableRouter._source_name(table_name)


def test_public_source_table_preserves_full_uri_for_connector_aware_validation(
) -> None:
    uri = 's3://company-data/nested/events.parquet'
    assert SourceTableRouter._source_name(uri) == uri


@pytest.mark.parametrize(
    ('config', 'table_name', 'data_source_type'),
    [
        (
            api.GCSConnectorResourceConfig(
                target=api.GCSAnyURIConnectorTarget()),
            'gs://company-data/nested/events.parquet',
            'GCS',
        ),
        (
            api.ADLSConnectorResourceConfig(
                target=api.ADLSAnyURIConnectorTarget()),
            ('abfss://curated@companylake.dfs.core.windows.net/'
             'nested/events.parquet'),
            'ADLS',
        ),
    ],
)
def test_public_source_table_config_accepts_connector_aware_any_uri(
        source_table_route, config, table_name, data_source_type) -> None:
    route = source_table_route(config)

    with TestClient(route.app) as client:
        response = client.get(
            f'/v1/projects/{route.project_id}/connectors/{route.connector_id}/'
            'source-table-config', params={'table_name': table_name})

    assert response.status_code == 200
    assert response.json()['source_table']['source'] == {
        'mode': 'uri',
        'uri': table_name,
    }
    assert response.json()['source_table']['data_source_type'] == (
        data_source_type)
    route.service.source_table_connector_for_project.assert_awaited_once_with(
        route.project_id, route.connector_id, 'principal')
    route.driver.get_project_source_table_file_type.assert_awaited_once_with(
        route.project_id, route.connector_id, 'principal', table_name)


def test_public_any_uri_provider_mismatch_is_rejected_after_authorization(
        source_table_route) -> None:
    config = api.GCSConnectorResourceConfig(
        target=api.GCSAnyURIConnectorTarget())
    route = source_table_route(config)

    with TestClient(route.app) as client:
        response = client.get(
            f'/v1/projects/{route.project_id}/connectors/{route.connector_id}/'
            'source-table-config',
            params={
                'table_name':
                ('abfss://curated@companylake.dfs.core.windows.net/'
                 'nested/events.parquet')
            },
        )

    assert response.status_code == 422
    route.service.source_table_connector_for_project.assert_awaited_once_with(
        route.project_id, route.connector_id, 'principal')
    route.driver.get_project_source_table_file_type.assert_not_awaited()


def test_public_source_table_relative_path_has_no_artificial_256_char_cap(
) -> None:
    table_name = '/'.join(['nested'] * 50 + ['events.parquet'])
    assert len(table_name) > 256
    assert SourceTableRouter._source_name(table_name) == table_name


@pytest.mark.parametrize(
    ('config', 'data_source_type'),
    [
        (
            api.S3ConnectorResourceConfig(target=api.S3RootedConnectorTarget(
                bucket='company-data', prefix='project-root')),
            'S3',
        ),
        (
            api.GCSConnectorResourceConfig(bucket='company-data',
                                           prefix='project-root'),
            'GCS',
        ),
        (
            api.ADLSConnectorResourceConfig(
                storage_account='companylake',
                filesystem='curated',
                prefix='project-root',
            ),
            'ADLS',
        ),
    ],
)
def test_v0_source_table_config_returns_exact_file_type_with_principal(
        config, data_source_type) -> None:
    project_id = ProjectID.generate()
    connector_id = ConnectorID.generate()
    principal = MagicMock()
    service = SimpleNamespace(source_table_connector_for_project=AsyncMock(
        return_value=SimpleNamespace(config=config)))
    driver = object.__new__(ConnectorApiDriver)
    driver._state_services = SimpleNamespace(connectors=service)
    driver.get_project_source_table_file_type = AsyncMock(
        return_value=(source_table_api.FileType.CSV, None))
    auth = SimpleNamespace(authenticate_request_header=lambda: None)
    resource = ConnectorResource(auth, driver)
    resource._principal = lambda _request: principal
    app = FastAPI()
    app.include_router(resource.router)

    table_name = 'nested/2026/events data'
    with TestClient(app) as client:
        response = client.get(
            f'/v0/projects/{project_id}/connectors/{connector_id}/'
            'source-table-config',
            params={'table_name': table_name},
        )

    assert response.status_code == 200
    expected = {
        'connector_id': str(connector_id),
        'file_type': 'CSV',
        'data_source_type': data_source_type,
    }
    expected['source'] = {'mode': 'relative', 'path': table_name}
    assert response.json()['source_table'] == expected
    service.source_table_connector_for_project.assert_awaited_once_with(
        project_id, connector_id, principal)
    driver.get_project_source_table_file_type.assert_awaited_once_with(
        project_id,
        connector_id,
        principal,
        table_name,
    )


def test_v0_source_table_config_rejects_traversal_before_resolution() -> None:
    project_id = ProjectID.generate()
    connector_id = ConnectorID.generate()
    driver = SimpleNamespace(get_project_source_table_config=AsyncMock())
    auth = SimpleNamespace(authenticate_request_header=lambda: None)
    resource = ConnectorResource(auth, driver)
    resource._principal = lambda _request: MagicMock()
    app = FastAPI()
    app.include_router(resource.router)

    with TestClient(app) as client:
        response = client.get(
            f'/v0/projects/{project_id}/connectors/{connector_id}/'
            'source-table-config',
            params={'table_name': 'nested/../events.parquet'},
        )

    assert response.status_code == 422
    driver.get_project_source_table_config.assert_not_awaited()


def test_source_download_transport_does_not_expand_ui_openapi() -> None:
    auth = SimpleNamespace(authenticate_request_header=lambda: None)
    resource = ConnectorResource(auth, MagicMock())
    app = FastAPI()
    app.include_router(resource.router)

    paths = app.openapi()['paths']
    assert not any(path.endswith('/download-urls') for path in paths)


@pytest.mark.parametrize(
    ('resource_type', 'url_prefix'),
    ((ConnectorResource, '/v0'), (ConnectorRouter, '/v1')),
)
@pytest.mark.asyncio
async def test_upload_start_uses_exact_project_and_scoped_part_url(
    resource_type,
    url_prefix,
) -> None:
    project_id = ProjectID.generate()
    connector_id = ConnectorID.generate()
    driver = SimpleNamespace(
        get_or_create_project_upload_connector=AsyncMock(
            return_value=connector_id),
        start_file_upload=AsyncMock(return_value=api.StartFileUploadResponse(
            temp_upload_path='/storage/temporary/file.csv',
            upload_id='upload-id',
            presigned_part_urls={
                1: '?temp_upload_path=/storage/temporary/file.csv'
            },
        )),
    )
    resource = object.__new__(resource_type)
    resource._driver = driver
    resource._principal = lambda _request: 'principal'

    result = await resource.start_file_upload(
        project_id,
        api.StartFileUploadRequest(
            source_table_name='orders',
            file_type=api.UploadFileType.CSV,
        ),
        object(),
    )

    driver.get_or_create_project_upload_connector.assert_awaited_once_with(
        project_id, 'principal')
    driver.start_file_upload.assert_awaited_once()
    assert driver.start_file_upload.await_args.args[:2] == (
        project_id,
        connector_id,
    )
    assert result.presigned_part_urls[1].startswith(
        f'{url_prefix}/projects/{project_id}/connectors/uploads/part?')


@pytest.mark.asyncio
async def test_project_upload_connector_authorizes_and_creates_scoped_root(
        monkeypatch) -> None:
    project_id = ProjectID.generate()
    connector_id = ConnectorID.generate()
    service = SimpleNamespace(
        require_project_access=AsyncMock(),
        register_project_upload_connector=AsyncMock(return_value=connector_id),
    )
    driver = object.__new__(ConnectorApiDriver)
    driver._state_services = SimpleNamespace(connectors=service)
    makedirs = MagicMock()
    monkeypatch.setattr(
        'rest.driver.connector_api_driver.deploy_env',
        lambda: SimpleNamespace(data_plane_storage_root='/storage'),
    )
    monkeypatch.setattr('rest.driver.connector_api_driver.fsutil.makedirs',
                        makedirs)

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr('rest.driver.connector_api_driver.asyncio.to_thread',
                        run_inline)

    assert await driver.get_or_create_project_upload_connector(
        project_id, 'principal') == connector_id

    service.require_project_access.assert_awaited_once_with(
        'principal', project_id)
    service.register_project_upload_connector.assert_awaited_once_with(
        project_id, f'/storage/uploaded-datasets/{project_id}')
    makedirs.assert_called_once_with(f'/storage/uploaded-datasets/{project_id}',
                                     exist_ok=True)


@pytest.mark.parametrize('resource_type', [ConnectorResource, ConnectorRouter])
@pytest.mark.asyncio
async def test_connector_get_shared_operation_preserves_project_and_principal(
        resource_type) -> None:
    project_id = ProjectID.generate()
    connector_id = ConnectorID.generate()
    expected = MagicMock(spec=api.ConnectorResponse)
    service = SimpleNamespace(get_for_project=AsyncMock(return_value=expected))
    driver = object.__new__(ConnectorApiDriver)
    driver._state_services = SimpleNamespace(connectors=service)
    resource = object.__new__(resource_type)
    resource._driver = driver
    resource._principal = lambda _request: 'principal'

    assert await resource.get_connector(project_id, connector_id,
                                        object()) is expected
    service.get_for_project.assert_awaited_once_with(project_id, connector_id,
                                                     'principal')


@pytest.mark.parametrize(
    ('resource_type', 'error'),
    [
        (
            ConnectorResource,
            NoIdentityValidatesTargetError(
                message='No available s3 identity could access this target.'),
        ),
        (
            ConnectorRouter,
            MultipleIdentitiesValidateTargetError(
                message=('Multiple identities can access this target; pass an '
                         'explicit identity: Production A (identity_safe).')),
        ),
        (
            ConnectorRouter,
            IdentityResolutionUnavailableError(
                message=('Connector identity resolution is temporarily '
                         'unavailable; please retry.')),
        ),
    ],
    ids=['v0-no-identity', 'v1-multiple-identities', 'v1-unavailable'],
)
@pytest.mark.asyncio
async def test_connector_create_preserves_sanitized_resolution_status(
        resource_type, error) -> None:
    resource = object.__new__(resource_type)
    resource._driver = SimpleNamespace(create_project_connector=AsyncMock(
        side_effect=error))
    resource._principal = lambda _request: 'principal'
    args = api.CreateConnectorArgs(
        name='events',
        config=api.S3ConnectorResourceConfig(
            target=api.S3RootedConnectorTarget(
                bucket='company-data',
                prefix='events',
                region='us-west-2',
            ), ),
    )

    with pytest.raises(type(error)) as caught:
        await resource.create_connector(ProjectID.generate(), args, object())

    assert caught.value.http_code == error.http_code
    assert str(caught.value) == str(error)
    assert 'provider-secret' not in str(caught.value)


@pytest.mark.parametrize(
    'resource_type',
    [ConnectorResource, ConnectorRouter],
    ids=['internal', 'public'],
)
def test_connector_create_name_conflict_remains_http_409(resource_type) -> None:
    project_id = ProjectID.generate()
    public_message = "Connector named 'warehouse' already exists in this project"
    private_details = (
        'credential=password-secret',
        'provider=databricks',
        'internal=unique-constraint-name',
    )
    error = ConnectorNameConflictError(public_message)
    error.__cause__ = RuntimeError(' '.join(private_details))
    driver = SimpleNamespace(create_project_connector=AsyncMock(
        side_effect=error))
    app = FastAPI()
    if resource_type is ConnectorResource:
        auth = SimpleNamespace(authenticate_request_header=lambda: None)
        resource = ConnectorResource(auth, driver)
        app.include_router(resource.router)
        url = f'/v0/projects/{project_id}/connectors/'
    else:
        resource = ConnectorRouter(lambda: None, driver)
        app.include_router(resource.router, prefix='/v1')
        url = f'/v1/projects/{project_id}/connectors'
    resource._principal = lambda _request: 'principal'

    with TestClient(app) as client:
        response = client.post(
            url,
            json={
                'name': 'warehouse',
                'config': {
                    'type': 'databricks',
                    'warehouse_id': 'warehouse-id',
                    'catalog': 'main',
                    'schema_name': 'features',
                },
            },
        )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json() == {'detail': public_message}
    assert all(detail not in response.text for detail in private_details)
    driver.create_project_connector.assert_awaited_once()


@pytest.mark.parametrize('operation', ['columns', 'sample'])
@pytest.mark.asyncio
async def test_missing_source_table_is_http_404(operation) -> None:
    error = ConnectorSourceTableNotFoundError('Unable to find table orders')
    driver = SimpleNamespace(
        get_project_source_columns=AsyncMock(side_effect=error),
        get_project_sample_rows=AsyncMock(side_effect=error),
    )
    resource = cast(Any, object.__new__(SourceTableRouter))
    resource._driver = driver
    resource._principal = lambda _request: 'principal'

    with pytest.raises(HTTPException) as caught:
        if operation == 'columns':
            await resource.get_columns(ProjectID.generate(),
                                       ConnectorID.generate(), object(),
                                       table_name='orders')
        else:
            await resource.get_sample(ProjectID.generate(),
                                      ConnectorID.generate(), object(),
                                      table_name='orders')

    assert caught.value.status_code == HTTPStatus.NOT_FOUND
    assert caught.value.detail == 'Unable to find table orders'


@pytest.mark.parametrize('resource_type', [ConnectorResource, ConnectorRouter])
@pytest.mark.asyncio
async def test_connector_create_hides_missing_explicit_identity_as_404(
        resource_type) -> None:
    identity_id = IdentityID.generate()
    resource = object.__new__(resource_type)
    resource._driver = SimpleNamespace(create_project_connector=AsyncMock(
        side_effect=IdentityDoesNotExistError(identity_id)))
    resource._principal = lambda _request: 'principal'
    args = api.CreateConnectorArgs(
        identity_id=identity_id,
        name='events',
        config=api.S3ConnectorResourceConfig(
            target=api.S3RootedConnectorTarget(
                bucket='company-data',
                prefix='events',
                region='us-west-2',
            ), ),
    )

    with pytest.raises(HTTPException) as caught:
        await resource.create_connector(ProjectID.generate(), args, object())

    assert caught.value.status_code == 404


@pytest.mark.parametrize('resource_type', [ConnectorResource, ConnectorRouter])
@pytest.mark.asyncio
async def test_stored_connector_validation_preserves_sanitized_422(
        resource_type) -> None:
    secret = 'provider-temporary-token-secret'
    error = ConnectorValidationError('Connector target validation failed')
    resource = object.__new__(resource_type)
    resource._driver = SimpleNamespace(
        validate_project_connector=AsyncMock(side_effect=error),
        get_project_connector=AsyncMock(),
    )
    resource._principal = lambda _request: 'principal'

    with pytest.raises(ConnectorValidationError) as caught:
        await resource.validate_connector(ProjectID.generate(),
                                          ConnectorID.generate(), object())

    assert caught.value.http_code == 422
    assert str(caught.value) == 'Connector target validation failed'
    assert secret not in str(caught.value)


def test_blank_databricks_schema_is_rejected_before_connector_activity():
    driver = SimpleNamespace(create_project_connector=AsyncMock())
    router = ConnectorRouter(lambda: None, driver)
    app = FastAPI()
    app.include_router(router.router, prefix='/v1')

    with TestClient(app) as client:
        response = client.post(
            f'/v1/projects/{ProjectID.generate()}/connectors',
            json={
                'name': 'warehouse',
                'config': {
                    'type': 'databricks',
                    'warehouse_id': 'warehouse-id',
                    'catalog': 'main',
                    'schema_name': '  ',
                },
            },
        )

    assert response.status_code == 422
    assert 'must not be blank' in response.text
    driver.create_project_connector.assert_not_awaited()


@pytest.mark.parametrize(
    ('endpoint', 'driver_method', 'activity_fn'),
    [
        ('columns', 'get_project_source_columns', get_connector_source_columns),
        ('sample', 'get_project_sample_rows', get_connector_sample_rows),
    ],
)
def test_connector_temporal_deadline_is_actionable_http_422(
        endpoint, driver_method, activity_fn) -> None:
    private_detail = 'provider-private-temporal-timeout'
    temporal_client = SimpleNamespace(execute_workflow=AsyncMock(
        side_effect=WorkflowFailureError(cause=TemporalTimeoutError(
            private_detail,
            type=TimeoutType.START_TO_CLOSE,
            last_heartbeat_details=(private_detail, ),
        ))))
    temporal = SimpleNamespace(get=AsyncMock(return_value=temporal_client))
    driver = object.__new__(ConnectorApiDriver)
    driver._data_backend_ctx = SimpleNamespace(temporal_client=lambda: temporal)

    async def timed_out_read(*_args, **_kwargs):
        return await driver._execute_connector_activity(
            activity_fn,
            [],
            timeout=120,
        )

    setattr(driver, driver_method, timed_out_read)
    resource = SourceTableRouter(lambda: None, driver,
                                 MagicMock(spec=Authentication))
    resource._principal = lambda _request: 'principal'
    app = FastAPI()
    app.include_router(resource.router, prefix='/v1')

    project_id = ProjectID.generate()
    connector_id = ConnectorID.generate()
    with TestClient(app) as client:
        response = client.get(
            f'/v1/projects/{project_id}/connectors/{connector_id}/{endpoint}',
            params={'table_name': 'events'},
        )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json() == {
        'detail':
        'Connector source read exceeded the configured 120-second budget. '
        'Set INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS to a larger value '
        'for this source, and keep any fronting proxy read timeout greater '
        'than the configured connector budget.'
    }
    assert private_detail not in response.text
    assert 'retry' not in response.text.lower()
    assert temporal_client.execute_workflow.await_args.kwargs[
        'execution_timeout'].total_seconds() == 120


@pytest.mark.parametrize(
    ('route', 'call_handler'),
    [
        ('source-tables', lambda r, p, c: r.list_source_tables(p, c, object())),
        ('columns',
         lambda r, p, c: r.get_columns(p, c, object(), table_name='orders')),
        ('sample',
         lambda r, p, c: r.get_sample(p, c, object(), table_name='orders')),
        ('source-table-config',
         lambda r, p, c: r.get_source_table_config(p, c, object(), 'orders')),
    ],
)
@pytest.mark.asyncio
async def test_discovery_maps_missing_connector_to_404(route,
                                                       call_handler) -> None:
    # A missing connector is a not-found, not an unprocessable request. All four
    # discovery routes must return 404 (matching the parent connector route),
    # not the 422 that a bare wrap_validation_exceptions() would produce for the
    # ConnectorDoesNotExistError (a ValueError subclass). Reverting the
    # not_exists_error argument on these handlers turns this back into a 422 and
    # fails here, so this test is a direct anti-regression guard.
    project_id = ProjectID.generate()
    connector_id = ConnectorID.generate()
    not_found = ConnectorDoesNotExistError(connector_id)
    resource = object.__new__(SourceTableRouter)
    resource._driver = SimpleNamespace(
        get_project_source_table_names=AsyncMock(side_effect=not_found),
        get_project_source_columns=AsyncMock(side_effect=not_found),
        get_project_sample_rows=AsyncMock(side_effect=not_found),
        get_project_source_table_config=AsyncMock(side_effect=not_found),
    )
    resource._principal = lambda _request: 'principal'

    with pytest.raises(HTTPException) as caught:
        await call_handler(resource, project_id, connector_id)

    assert caught.value.status_code == 404, route
    assert 'does not exist' in str(caught.value.detail)


@pytest.mark.parametrize(
    ('route', 'call_handler'),
    [
        ('source-tables', lambda r, p, c: r.list_source_tables(p, c, object())),
        ('columns',
         lambda r, p, c: r.get_columns(p, c, object(), table_name='orders')),
        ('sample',
         lambda r, p, c: r.get_sample(p, c, object(), table_name='orders')),
        ('source-table-config',
         lambda r, p, c: r.get_source_table_config(p, c, object(), 'orders')),
    ],
)
@pytest.mark.asyncio
async def test_internal_discovery_maps_missing_connector_to_404(
        route, call_handler) -> None:
    # A missing connector is a not-found, not an unprocessable request. All
    # four internal discovery routes must return 404, not the 422 that a bare
    # wrap_validation_exceptions() would produce for the
    # ConnectorDoesNotExistError (a ValueError subclass), and must stay
    # consistent with the public discovery routes (see #29816 / #29819).
    # Reverting the not_exists_error argument on these handlers turns this
    # back into a 422 and fails here.
    project_id = ProjectID.generate()
    connector_id = ConnectorID.generate()
    not_found = ConnectorDoesNotExistError(connector_id)
    resource = object.__new__(ConnectorResource)
    resource._driver = SimpleNamespace(
        get_project_source_table_names=AsyncMock(side_effect=not_found),
        get_project_source_columns=AsyncMock(side_effect=not_found),
        get_project_sample_rows=AsyncMock(side_effect=not_found),
        get_project_source_table_config=AsyncMock(side_effect=not_found),
    )
    resource._principal = lambda _request: 'principal'

    with pytest.raises(HTTPException) as caught:
        await call_handler(resource, project_id, connector_id)

    assert caught.value.status_code == 404, route
    assert 'does not exist' in str(caught.value.detail)
