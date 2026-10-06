# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import os
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import kumoapi.data_snapshot as snapshot_interfaces
import kumoapi.data_source as data_source_interfaces
import kumoapi.source_table as source_table_interfaces
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from kumoapi.data_source import DEFAULT_PROJECT_ID
from pydantic import SecretStr
from temporalio import activity

from io_utils import fsutil
from io_utils.json import load_json
from kumo.connector.snowflake_connector import SnowflakeConnector
from kumo.data import get_active_cache
from kumo.data.backend_cache import BackendCacheManager
from kumo.testing.rest import (
    get_and_expect_error,
    get_and_expect_success,
    post_and_expect_success,
)
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.data_backend import DataBackend
from rest.config.graph_v2 import GraphDefinition
from rest.config.predictive_query_v2 import PQueryResource
from rest.config.status import IDResponse
from rest.deps.authentication import UserAuthorization
from rest.testing import integration
from service_framework.deploy_env_config import DeployEnvConfig, deploy_env
from statedb.config.common import ConnectorID, DataSnapshot, ProjectID
from statedb.rest.connector_authorization import CONNECTOR_GRAPH_ACTIONS
from temporalio_commons.temporal_client import TemporalClient

_FINANCIAL_DATA_PATH = (Path(__file__).resolve().parents[2] / 'csv_data' /
                        'FINANCIAL')


async def _terminate_and_confirm_no_open_workflows(
    config: DeployEnvConfig, ) -> None:
    await integration.terminate_and_confirm_no_open_workflows(config)


def _project_id(client: TestClient) -> str:
    del client
    return DEFAULT_PROJECT_ID


def _stage_project_csvs(
    client: TestClient,
    project_id: str,
    pquery_res: PQueryResource,
) -> None:
    for table in pquery_res.graph.tables.values():
        source = table.source_table.source
        assert isinstance(source, source_table_interfaces.S3RelativeSource)
        source_filename = str(source.path)
        source_path = _FINANCIAL_DATA_PATH / source_filename
        source_table_name = source_path.stem
        start = client.post(
            f'/v1/projects/{project_id}/connectors/uploads/start',
            json={
                'source_table_name': source_table_name,
                'file_type': 'CSV',
                'num_parts': 1,
            },
        )
        assert start.status_code == HTTPStatus.OK, start.text
        upload = start.json()
        part = client.put(
            upload['presigned_part_urls']['1'],
            content=source_path.read_bytes(),
        )
        assert part.status_code == HTTPStatus.OK, part.text
        complete = client.post(
            f'/v1/projects/{project_id}/connectors/uploads/complete',
            json={
                'source_table_name':
                source_table_name,
                'file_type':
                'CSV',
                'temp_upload_path':
                upload['temp_upload_path'],
                'upload_id':
                upload['upload_id'],
                'parts_metadata': [{
                    'part_number': 1,
                    'etag': part.json()['etag'],
                }],
                'validate_data':
                False,
            },
        )
        assert complete.status_code == HTTPStatus.OK, complete.text
    connector = client.get(
        f'/v1/projects/{project_id}/connectors/uploads/connector')
    assert connector.status_code == HTTPStatus.OK, connector.text
    connector_id = connector.json()['connector_id']
    for table_name, table in list(pquery_res.graph.tables.items()):
        source = table.source_table.source
        assert isinstance(source, source_table_interfaces.S3RelativeSource)
        source_table = dataclasses.replace(
            table.source_table,
            source=dataclasses.replace(source,
                                       path=Path(str(source.path)).stem),
            connector_id=connector_id,
        )
        pquery_res.graph.tables[table_name] = dataclasses.replace(
            table, source_table=source_table)


async def _create_snapshot_snowflake_connector(
    state_services,
    monkeypatch,
):
    admin = UserAuthorization.disabled()
    project_id = ProjectID(DEFAULT_PROJECT_ID)
    project = await state_services.projects.get_project(project_id, admin)
    monkeypatch.setattr(state_services.identities, '_identity_validator',
                        lambda *_args: None)
    monkeypatch.setattr(state_services.connectors, '_connector_validator',
                        lambda *_args: None)
    identity = await state_services.identities.create(
        data_source_interfaces.CreateIdentityArgs(
            display_name='Snapshot Snowflake identity',
            config=data_source_interfaces.SnowflakeIdentityConfig(
                account='snapshot-organization-account',
                username='SNAPSHOT_READER',
                authentication=(data_source_interfaces.
                                SnowflakeAuthenticationKind.PASSWORD),
            ),
            credentials=(data_source_interfaces.SnowflakePasswordCredentials(
                password=SecretStr('snapshot-snowflake-secret'))),
        ),
        admin,
    )
    await state_services.identities.replace_groups(
        identity.id,
        data_source_interfaces.ReplaceIdentityGroupsArgs(
            group_ids=[str(project.group_id)]),
        admin,
    )
    args = data_source_interfaces.CreateConnectorArgs(
        identity_id=str(identity.id),
        name='Snapshot Snowflake connector',
        config=data_source_interfaces.SnowflakeConnectorResourceConfig(
            role='SNAPSHOT_READER',
            warehouse='COMPUTE_WH',
            database='PRODUCTION',
            schema_name='ANALYTICS',
        ),
    )
    authorizations = await state_services.connectors.authorize_create(
        project_id, args, admin)
    connector = await state_services.connectors.create_from_activity(
        project_id,
        args,
        admin.user_id,
        set(authorizations),
    )
    return admin, project, identity, connector


def _snapshot_snowflake_graph(connector_id: str) -> dict:
    return {
        'tables': {
            'events': {
                'cols': [{
                    'name': 'entity_id',
                    'stype': 'ID',
                    'dtype': 'int',
                }, {
                    'name': 'label',
                    'stype': 'categorical',
                    'dtype': 'string',
                }],
                'source_table': {
                    'snowflake_connector_id': connector_id,
                    'database': 'PRODUCTION',
                    'schema_name': 'ANALYTICS',
                    'table': 'events',
                    'data_source_type': 'SNOWFLAKE',
                },
                'pkey':
                'entity_id',
                'time_col':
                None,
            },
        },
        'col_groups': [],
    }


@pytest_asyncio.fixture(scope="function", autouse=True)
async def setup_and_teardown_test_func(public_api_client: TestClient,
                                       deploy_env_config: DeployEnvConfig):
    await _terminate_and_confirm_no_open_workflows(deploy_env_config)
    # ``reset()`` clears the sub-caches but leaves the cache dir untouched, so
    # the tmpdir override set by the session-scoped
    # ``force_local_cache_dir_for_public_tests`` fixture is preserved.
    cache = get_active_cache()
    cache.reset()
    yield
    cache = get_active_cache()
    cache.reset()
    await _terminate_and_confirm_no_open_workflows(deploy_env_config)


@pytest.fixture(scope='module', autouse=True)
def setup_teardown_test_module(data_backend_ctx: DataBackendContext):
    data_backend_settings = data_backend_ctx._settings
    disable_async_ingestion = data_backend_settings.disable_async_ingestion
    data_backend_settings.disable_async_ingestion = False
    # Data backend cache: reset in-memory cache; backup and reset persisted
    # cache dir and restore after test module.
    cache: BackendCacheManager = get_active_cache()

    with TemporaryDirectory(ignore_cleanup_errors=True) as cache_backup_dir:
        cache.reset()
        fsutil.copy(cache.cache_dir, cache_backup_dir)
        fsutil.rm(cache.cache_dir, ignore_errors=True)
        yield
        data_backend_settings.disable_async_ingestion = disable_async_ingestion
        cache.reset()
        fsutil.rm(cache.cache_dir, ignore_errors=True)
        fsutil.copy(cache_backup_dir, cache.cache_dir, overwrite=True,
                    raise_on_exists=False)


@pytest.fixture(scope="module", autouse=True)
def temporal_workers_ready(
        deploy_env_config: DeployEnvConfig) -> Generator[bool, None, None]:
    # setup_rest_test_session replaces the process-global test config after it
    # installs the isolated Kumo home. Workers must use that active config so
    # activity-side connector resolution opens the same StateDB as REST.
    active_config = deploy_env()
    previous_disable_auth = os.environ.get('DISABLE_AUTH')
    os.environ['DISABLE_AUTH'] = 'true'
    workers = integration.run_temporal_workers(active_config)
    ready = next(workers)
    try:
        yield ready
    finally:
        asyncio.run(_terminate_and_confirm_no_open_workflows(active_config))
        try:
            next(workers)
        except StopIteration:
            pass
        else:
            raise AssertionError('Temporal worker fixture did not stop')
        if previous_disable_auth is None:
            os.environ.pop('DISABLE_AUTH', None)
        else:
            os.environ['DISABLE_AUTH'] = previous_disable_auth


@pytest.fixture(scope="function")
def pquery_res() -> PQueryResource:
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    test_data_dir = os.path.join(current_dir.parent, 'test_data')
    pq_res_json_file = os.path.join(test_data_dir, 'pquery_resource.json')
    return load_json(pq_res_json_file, PQueryResource)


@pytest.mark.asyncio
async def test_graph_snapshot(public_api_client: TestClient,
                              temporal_workers_ready: bool,
                              pquery_res: PQueryResource,
                              deploy_env_config: DeployEnvConfig, caplog):
    project_id = _project_id(public_api_client)
    get_and_expect_error(public_api_client, 'v1/graphsnapshots/badsnapshotid', {
        'project_id': project_id,
    }, HTTPStatus.UNPROCESSABLE_ENTITY, 'GRAPH snapshot ID is invalid',
                         skip_cache=False)

    _stage_project_csvs(public_api_client, project_id, pquery_res)

    resp = post_and_expect_success(public_api_client,
                                   f'/v1/graphs?project_id={project_id}',
                                   pquery_res.graph, IDResponse)
    graph_id = resp.id

    resp = post_and_expect_success(
        public_api_client, f'/v1/graphsnapshots?graph_id={graph_id}'
        f'&project_id={project_id}', {}, IDResponse, skip_cache=False)
    graph_snapshot_id = resp.id
    graph_fpid, _ = DataSnapshot.GRAPH.parse_id(graph_snapshot_id)
    graph_def = GraphDefinition(**dataclasses.asdict(pquery_res.graph))
    assert graph_def.fpid == graph_fpid

    snapshot = get_and_expect_success(public_api_client,
                                      f'v1/graphsnapshots/{graph_snapshot_id}',
                                      {'project_id': project_id},
                                      snapshot_interfaces.GraphSnapshotResource,
                                      skip_cache=False)
    assert snapshot.id == graph_snapshot_id
    assert set(snapshot.stages) == {
        snapshot_interfaces.GraphSnapshotStage.INGEST
    }
    ingest_state = snapshot.stages[
        snapshot_interfaces.GraphSnapshotStage.INGEST]
    assert ingest_state.status in {
        snapshot_interfaces.JobStatus.RUNNING,
        snapshot_interfaces.JobStatus.DONE,
    }
    assert ingest_state.error is None
    await _terminate_and_confirm_no_open_workflows(deploy_env_config)
    assert not any('Authentication mode changed' in record.getMessage()
                   for record in caplog.records)


def test_identity_bound_graph_snapshot_capability_boundary(
    public_api_client: TestClient,
    temporal_workers_ready: bool,
    state_services,
    data_backend_ctx: DataBackendContext,
    monkeypatch,
):
    del temporal_workers_ready
    admin, project, identity, connector = asyncio.run(
        _create_snapshot_snowflake_connector(state_services, monkeypatch))
    project_id = ProjectID(DEFAULT_PROJECT_ID)
    provider_secret = 'snapshot-snowflake-secret'

    def require_activity_boundary():
        assert activity.in_activity(), 'request-side Snowflake provider I/O'

    open_provider = MagicMock(side_effect=lambda *_args, **_kwargs:
                              (require_activity_boundary(), object())[1])
    get_data_version = MagicMock(side_effect=lambda *_args, **_kwargs: (
        require_activity_boundary(), 'snapshot-snowflake-version')[1])
    close_provider = MagicMock(
        side_effect=lambda *_args, **_kwargs: require_activity_boundary())
    monkeypatch.setattr(SnowflakeConnector, '_open', open_provider)
    monkeypatch.setattr(SnowflakeConnector, '_get_data_version',
                        get_data_version)
    monkeypatch.setattr(SnowflakeConnector, '_close', close_provider)
    monkeypatch.setattr(SnowflakeConnector, 'validate_permissions',
                        lambda _self: None)

    # This test exercises the version-resolution workflow, not downstream
    # ingestion/materialization workflows.
    monkeypatch.setattr(DataBackend, 'get_many_table_refresh_results',
                        AsyncMock(return_value=[None]))
    monkeypatch.setattr(DataBackend, 'start_refresh_table', AsyncMock())
    monkeypatch.setattr(DataBackend, 'start_refresh_graph', AsyncMock())

    issued_references = []
    issue_reference = state_services.connectors.execution_ref_for_project

    async def capture_reference(*args, **kwargs):
        reference = await issue_reference(*args, **kwargs)
        issued_references.append(reference)
        return reference

    monkeypatch.setattr(state_services.connectors, 'execution_ref_for_project',
                        capture_reference)

    from rest.backends.data_backend import data_backend as data_backend_module
    uuid_hexes = []

    def deterministic_uuid():
        value = f'{len(uuid_hexes) + 1:032x}'
        uuid_hexes.append(value)
        return SimpleNamespace(hex=value)

    monkeypatch.setattr(data_backend_module, 'uuid4', deterministic_uuid)
    execute_activity = DataBackend._execute_single_activity
    workflow_ids = []

    async def track_secret_free_activity(ctx, activity_fn, inputs, prefix):
        assert provider_secret not in repr(inputs)
        uuid_index = len(uuid_hexes)
        try:
            result = await execute_activity(ctx, activity_fn, inputs, prefix)
            assert provider_secret not in repr(result)
            return result
        finally:
            if len(uuid_hexes) > uuid_index:
                workflow_ids.append(f'{prefix}:{uuid_hexes[uuid_index]}')

    monkeypatch.setattr(DataBackend, '_execute_single_activity',
                        track_secret_free_activity)

    graph_response = public_api_client.post(
        '/v1/graphs',
        params={
            'project_id': DEFAULT_PROJECT_ID,
            'skip_validation': 'true',
        },
        json=_snapshot_snowflake_graph(str(connector.id)),
    )
    assert graph_response.status_code == HTTPStatus.OK, graph_response.text
    graph_id = graph_response.json()['id']
    issued_references.clear()
    open_provider.reset_mock()
    get_data_version.reset_mock()
    close_provider.reset_mock()

    snapshot_response = public_api_client.post(
        '/v1/graphsnapshots',
        params={
            'project_id': DEFAULT_PROJECT_ID,
            'graph_id': graph_id,
            'refresh_source': 'true',
        },
        json={},
    )
    assert snapshot_response.status_code == HTTPStatus.OK, (
        snapshot_response.text)
    assert open_provider.call_count == 1
    assert get_data_version.call_count == 1
    assert close_provider.call_count == 1

    assert len(issued_references) == 4
    assert len(
        {reference.authorization_token
         for reference in issued_references}) == 4
    table_version_reference, table_refresh_reference, \
        graph_version_reference, graph_refresh_reference = issued_references
    assert table_version_reference.connector_id == (
        table_refresh_reference.connector_id)
    assert graph_version_reference.connector_id == (
        graph_refresh_reference.connector_id)

    successful_reference = issued_references[0]
    table_version_workflow_id = next(
        workflow_id for workflow_id in workflow_ids
        if workflow_id.startswith('table-data-version:'))
    graph_version_workflow_id = next(
        workflow_id for workflow_id in workflow_ids
        if workflow_id.startswith('refresh-graph-version:'))

    async def read_authorizations():
        async with state_services.state_db.read_tx() as tx:
            return tuple([
                await
                state_services.connector_authorizations._get_by_token_in_tx(
                    tx, reference.authorization_token)
                for reference in issued_references
            ])

    table_version_auth, table_refresh_auth, graph_version_auth, \
        graph_refresh_auth = asyncio.run(read_authorizations())
    expected_actions = tuple(
        sorted(CONNECTOR_GRAPH_ACTIONS, key=lambda action: action.value))
    assert table_version_auth.allowed_actions == expected_actions
    assert table_version_auth.workflow_id == deploy_env().workflow_id(
        table_version_workflow_id)
    assert table_refresh_auth.allowed_actions == expected_actions
    assert table_refresh_auth.workflow_id == ''
    assert graph_version_auth.allowed_actions == expected_actions
    assert graph_version_auth.workflow_id == deploy_env().workflow_id(
        graph_version_workflow_id)
    assert graph_refresh_auth.allowed_actions == expected_actions
    assert graph_refresh_auth.workflow_id == ''

    async def read_history() -> bytes:
        client = await TemporalClient(deploy_env()).get()
        handle = client.get_workflow_handle(
            deploy_env().workflow_id(table_version_workflow_id))
        events = [event async for event in handle.fetch_history_events()]
        return b''.join(event.SerializeToString() for event in events)

    history = asyncio.run(read_history())
    for marker in (provider_secret.encode(), b'SNAPSHOT_READER',
                   b'snapshot-organization-account'):
        assert marker not in history

    # A capability is bound to the first workflow/run and exact project.
    with pytest.raises(Exception):
        asyncio.run(
            DataBackend.get_table_data_version_in_activity(
                data_backend_ctx,
                successful_reference,
                'events',
                refresh_table_version=True,
            ))
    cross_project_reference = dataclasses.replace(
        successful_reference,
        project_id=ProjectID('project_' + 'c' * 32),
    )
    with pytest.raises(Exception):
        asyncio.run(
            DataBackend.get_table_data_version_in_activity(
                data_backend_ctx,
                cross_project_reference,
                'events',
                refresh_table_version=True,
            ))

    # An unused, otherwise valid capability must recheck the current binding
    # inside the activity. Removing the group binding denies before provider
    # open/version/close and also prevents a fresh route invocation.
    unused_reference = asyncio.run(
        issue_reference(
            admin,
            project_id,
            ConnectorID(connector.id),
            CONNECTOR_GRAPH_ACTIONS,
        ))
    asyncio.run(
        state_services.identities.replace_groups(
            identity.id,
            data_source_interfaces.ReplaceIdentityGroupsArgs(group_ids=[]),
            admin,
        ))
    with pytest.raises(Exception):
        asyncio.run(
            DataBackend.get_table_data_version_in_activity(
                data_backend_ctx,
                unused_reference,
                'events',
                refresh_table_version=True,
            ))
    denied_response = public_api_client.post(
        '/v1/graphsnapshots',
        params={
            'project_id': DEFAULT_PROJECT_ID,
            'graph_id': graph_id,
            'refresh_source': 'true',
        },
        json={},
    )
    assert denied_response.status_code != HTTPStatus.OK
    assert provider_secret not in denied_response.text
    assert open_provider.call_count == 1
    assert get_data_version.call_count == 1
    assert close_provider.call_count == 1
    assert project.group_id is not None
