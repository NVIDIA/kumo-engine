# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import json
import subprocess
import sys
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import kumoapi.data_source as api
import pytest
from kumo_core.types import ConnectorType
from temporalio.client import WorkflowExecutionStatus, WorkflowFailureError
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    RetryState,
    TimeoutError,
    TimeoutType,
)

from common_utils.error import PublicException
from kumo.connector.composite_file_connector import CompositeFileConnector
from kumo.connector.connector import (
    ConnectorError,
    SourceTableNotFoundError,
    SourceTableUnavailableError,
)
from kumo.connector.databricks_connector import DatabricksConnector
from kumo.connector.file_connector import FileConnector
from kumo.validation import ValidationException
from rest.backends.data_backend.activities.identity_management import (
    create_project_connector_activity,
    validate_identity_credentials_activity,
)
from rest.config.connector import SourceTablesResponse
from rest.config.data_backend import (
    ActivityAuthorizationRef,
    ConnectorExecutionRef,
    ConnectorSpec,
    ResolvedConnectorSpec,
)
from rest.driver.connector_api_driver import (
    _INTERACTIVE_CONNECTOR_READ_TIMEOUT_ENV,
    _INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS,
    ConnectorApiDriver,
    _get_source_tables_raw,
    get_connector_sample_rows,
    get_connector_source_columns,
    get_connector_source_table_file_type,
    is_connector_source_table_view,
    list_connector_source_tables,
    stage_connector_source_download,
    validate_connector_reference,
    validate_connector_source_table_exists,
)
from statedb.config.common import ConnectorID, IdentityID, ProjectID, UserID
from statedb.rest.connector_authorization import CONNECTOR_DOWNLOAD_ACTIONS
from statedb.utils.errors import (
    ConnectorNameConflictError,
    ConnectorSourceTableNotFoundError,
    ConnectorValidationError,
    CustomSqlQueryError,
    IdentityResolutionUnavailableError,
    IdentityValidationError,
    MultipleIdentitiesValidateTargetError,
    NoIdentityValidatesTargetError,
    ResourceSlugConflictError,
)
from temporalio_commons.auto_rpc_exception import RpcError
from temporalio_commons.worker_registry import RunSingleActivityWorkflow

PROJECT_ID = ProjectID('project_' + '1' * 32)
CONNECTOR_ID = ConnectorID('connector_' + '2' * 32)
IDENTITY_ID = IdentityID('identity_' + '3' * 32)
REFERENCE = ConnectorExecutionRef(project_id=PROJECT_ID,
                                  connector_id=CONNECTOR_ID)
AUTHORIZATION = ActivityAuthorizationRef('opaque-test-capability')
CREATE_AUTHORIZATIONS = {IDENTITY_ID: AUTHORIZATION}
_BROKER_RESOLVE = ('rest.backends.data_backend.connector_broker.'
                   'connector_credential_broker.resolve')
_CONNECTOR_FACTORY = 'rest.driver.connector_api_driver.get_connector_from_spec'


def _resolved_external_spec() -> ResolvedConnectorSpec:
    return ResolvedConnectorSpec(
        connector_id=CONNECTOR_ID,
        connector_config=MagicMock(),
        credentials=None,
    )


def _resolved_internal_spec(*, is_project_upload: bool) -> ConnectorSpec:
    return ConnectorSpec(
        connector_id=CONNECTOR_ID,
        connector_config=MagicMock(),
        is_project_upload=is_project_upload,
    )


def _driver(result):
    client = SimpleNamespace(execute_workflow=AsyncMock(return_value=result))
    temporal = SimpleNamespace(get=AsyncMock(return_value=client))
    context = SimpleNamespace(temporal_client=lambda: temporal)
    connectors = SimpleNamespace(
        execution_ref_for_project=AsyncMock(return_value=REFERENCE),
        authorize_create=AsyncMock(return_value=CREATE_AUTHORIZATIONS),
        get_for_project=AsyncMock(),
        source_table_connector_for_project=AsyncMock(return_value=None),
    )
    identities = SimpleNamespace(authorize_test_credentials=AsyncMock(
        return_value=AUTHORIZATION))
    tables = SimpleNamespace(list_tables=AsyncMock(return_value=[]))
    driver = ConnectorApiDriver(
        context,
        list(ConnectorType),
        state_services=SimpleNamespace(
            connectors=connectors,
            identities=identities,
            tables=tables,
        ),
    )
    return driver, connectors, identities, client


def _databricks_create_args(identity_id=IDENTITY_ID):
    return api.CreateConnectorArgs(
        identity_id=str(identity_id) if identity_id is not None else None,
        name='warehouse',
        config=api.DatabricksConnectorResourceConfig(
            warehouse_id='warehouse-id',
            catalog='main',
            schema_name='features',
        ),
    )


def _assert_sanitized_internal_error(error: RpcError, secret: str) -> None:
    assert error.error_type == 'InternalError'
    assert secret not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def _activity_response_token(activity_fn) -> str:
    return f'connector-operation:{activity_fn.__name__}:{"1" * 32}'


def _isolated_name_conflict_payload(message: str) -> dict:
    script = (
        'import json\n'
        'import sys\n'
        'from statedb.utils.errors import ResourceSlugConflictError\n'
        "impostor_type = type('ConnectorNameConflictError', "
        '(ResourceSlugConflictError,), {})\n'
        'print(json.dumps(impostor_type(message=sys.argv[1]).to_dict()))\n')
    result = subprocess.run(
        [sys.executable, '-c', script, message],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _run_activity_service(services, authorization, authorize_method, invoke):
    def run(operation):
        return asyncio.run(operation(services))

    broker_path = ('rest.backends.data_backend.activities.identity_management.'
                   'connector_credential_broker')
    with patch(
            f'{broker_path}.{authorize_method}',
            return_value=authorization,
    ), patch(f'{broker_path}.run', side_effect=run) as broker:
        return invoke(), broker


def _assert_connector_failure_is_sanitized(invoke, secret, caplog=None):
    with pytest.raises(ConnectorValidationError) as caught:
        invoke()
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    if caplog is not None:
        assert secret not in caplog.text
    return caught.value


@pytest.mark.asyncio
async def test_prediction_sdk_connector_metadata_uses_authorized_safe_name():
    driver, connectors, _identities, _client = _driver(None)
    principal = MagicMock()
    connectors.get_for_project.return_value = SimpleNamespace(
        id=CONNECTOR_ID,
        name='Shared output',
        config=SimpleNamespace(type=api.ConnectorType.S3),
    )

    result = await driver.get_bp_connector_metadata_for_sdk(
        {
            'connector_id': str(CONNECTOR_ID),
            'data_source_type': 'S3',
        },
        PROJECT_ID,
        principal,
    )

    connectors.get_for_project.assert_awaited_once_with(
        PROJECT_ID,
        CONNECTOR_ID,
        principal,
    )
    assert result == {
        'connector_id': str(CONNECTOR_ID),
        'lookup_name': 'Shared output',
        'data_source_type': 'S3',
    }


@pytest.mark.asyncio
async def test_prediction_sdk_connector_metadata_rejects_wrong_type():
    driver, connectors, _identities, _client = _driver(None)
    connectors.get_for_project.return_value = SimpleNamespace(
        id=CONNECTOR_ID,
        name='Shared output',
        config=SimpleNamespace(type=api.ConnectorType.GCS),
    )

    with pytest.raises(ValueError, match="has type 'GCS', not 'S3'"):
        await driver.get_bp_connector_metadata_for_sdk(
            {
                'connector_id': str(CONNECTOR_ID),
                'data_source_type': 'S3',
            },
            PROJECT_ID,
            MagicMock(),
        )


@pytest.mark.asyncio
async def test_source_table_discovery_sends_only_execution_reference():
    expected = SourceTablesResponse()
    driver, connectors, _identities, client = _driver(expected)

    result = await driver.get_project_source_tables(PROJECT_ID, CONNECTOR_ID,
                                                    MagicMock(), sync=True)

    assert result == expected
    connectors.execution_ref_for_project.assert_awaited_once()
    call = client.execute_workflow.await_args
    assert call.args[0] is RunSingleActivityWorkflow.run
    assert call.kwargs['args'][0] == list_connector_source_tables.__name__
    activity_inputs = call.kwargs['args'][1]
    assert activity_inputs[0] == REFERENCE
    assert activity_inputs == [REFERENCE, [], None]
    assert all(not hasattr(value, 'credentials') for value in activity_inputs)


@pytest.mark.asyncio
async def test_source_download_url_is_created_after_temporal_result():
    internal_path = 's3://example-internal/temporary/table'
    driver, connectors, _identities, client = _driver(internal_path)
    presign = AsyncMock(return_value=['https://signed/internal'])

    with patch('rest.driver.connector_api_driver.presign_parquet_table_data',
               SimpleNamespace(coroutine=presign)):
        result = await driver.get_project_source_download_urls(
            PROJECT_ID, CONNECTOR_ID, MagicMock(), 'orders')

    assert result == ['https://signed/internal']
    assert client.execute_workflow.await_args.kwargs['args'][0] == (
        stage_connector_source_download.__name__)
    assert internal_path not in client.execute_workflow.await_args.kwargs[
        'args'][1]
    presign.assert_awaited_once_with(internal_path)
    actions = connectors.execution_ref_for_project.await_args.args[-1]
    assert actions == CONNECTOR_DOWNLOAD_ACTIONS


@pytest.mark.asyncio
async def test_source_download_requests_use_immutable_staging_ids():
    driver, _connectors, _identities, client = _driver(None)
    client.execute_workflow.side_effect = [
        's3://example-internal/downloads/first/table',
        's3://example-internal/downloads/second/table',
    ]
    presign = AsyncMock(
        side_effect=[['https://signed/first'], ['https://signed/second']])

    with patch('rest.driver.connector_api_driver.presign_parquet_table_data',
               SimpleNamespace(coroutine=presign)):
        first = await driver.get_project_source_download_urls(
            PROJECT_ID, CONNECTOR_ID, MagicMock(), 'orders')
        second = await driver.get_project_source_download_urls(
            PROJECT_ID, CONNECTOR_ID, MagicMock(), 'orders')

    calls = client.execute_workflow.await_args_list
    first_id = calls[0].kwargs['args'][1][-1]
    second_id = calls[1].kwargs['args'][1][-1]
    assert (first, second) == (['https://signed/first'],
                               ['https://signed/second'])
    assert first_id != second_id
    assert len(first_id) == len(second_id) == 32


@pytest.mark.asyncio
async def test_source_columns_query_arg_only_sent_in_query_mode():
    """The physical columns call keeps its prior 3-arg activity arity so a
    rolling deploy to an old worker (no ``query`` parameter) still works; a
    Custom SQL ``query`` appends the new arg."""
    driver, _connectors, _identities, client = _driver(SimpleNamespace())

    await driver.get_project_source_columns(PROJECT_ID, CONNECTOR_ID,
                                            MagicMock(), 'orders')
    physical_inputs = client.execute_workflow.await_args.kwargs['args'][1]
    assert physical_inputs == [REFERENCE, 'orders', True]

    await driver.get_project_source_columns(PROJECT_ID, CONNECTOR_ID,
                                            MagicMock(), '',
                                            query='SELECT a FROM t')
    query_inputs = client.execute_workflow.await_args.kwargs['args'][1]
    assert query_inputs == [REFERENCE, '', True, 'SELECT a FROM t']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('config', 'source_name', 'normalized_source_name'),
    [
        (
            api.SnowflakeConnectorResourceConfig(
                warehouse='COMPUTE_WH',
                database='PROD',
                schema_name='SALES',
            ),
            'orders',
            'orders',
        ),
        (
            api.DatabricksConnectorResourceConfig(
                warehouse_id='warehouse',
                catalog='main',
                schema_name='sales',
            ),
            'main.sales.订单@#$%20?__kumo__table',
            'sales.订单@#$%20?__kumo__table',
        ),
        (
            api.BigQueryConnectorResourceConfig(
                project_id='analytics',
                dataset_id='sales',
                location='US',
            ),
            'orders',
            'orders',
        ),
    ],
    ids=['snowflake', 'databricks', 'bigquery'],
)
async def test_warehouse_exact_config_reauthorizes_and_validates_each_call(
        config, source_name, normalized_source_name):
    driver, connectors, _identities, client = _driver(True)
    principal = MagicMock()
    connectors.source_table_connector_for_project.return_value = (
        SimpleNamespace(config=config))

    await driver.get_project_source_table_config(PROJECT_ID, CONNECTOR_ID,
                                                 principal, source_name)
    await driver.get_project_source_table_config(PROJECT_ID, CONNECTOR_ID,
                                                 principal, source_name)

    assert connectors.source_table_connector_for_project.await_count == 2
    assert connectors.execution_ref_for_project.await_count == 2
    assert client.execute_workflow.await_count == 2
    for activity_call in client.execute_workflow.await_args_list:
        assert activity_call.kwargs['args'] == [
            validate_connector_source_table_exists.__name__,
            [REFERENCE, normalized_source_name],
        ]


@pytest.mark.asyncio
async def test_databricks_exact_config_rejects_out_of_scope_coordinate_before_io(
):
    driver, connectors, _identities, client = _driver(True)
    principal = MagicMock()
    connectors.source_table_connector_for_project.return_value = (
        SimpleNamespace(config=api.DatabricksConnectorResourceConfig(
            warehouse_id='warehouse',
            catalog='main',
            schema_name='sales',
        )))

    coordinate, _ = await driver._validated_source_coordinate(
        PROJECT_ID,
        CONNECTOR_ID,
        principal,
        'literal__kumo__table',
    )
    assert coordinate == 'sales.literal__kumo__table'

    with pytest.raises(ConnectorError, match='connector schema'):
        await driver.get_project_source_table_config(
            PROJECT_ID,
            CONNECTOR_ID,
            principal,
            'main.private.orders',
        )

    client.execute_workflow.assert_not_awaited()


@pytest.mark.asyncio
async def test_warehouse_exact_config_rejects_missing_table():
    driver, connectors, _identities, client = _driver(False)
    connectors.source_table_connector_for_project.return_value = (
        SimpleNamespace(config=api.SnowflakeConnectorResourceConfig(
            warehouse='COMPUTE_WH', database='PROD', schema_name='SALES')))

    with pytest.raises(ValueError, match='Invalid table orders'):
        await driver.get_project_source_table_config(PROJECT_ID, CONNECTOR_ID,
                                                     MagicMock(), 'orders')

    assert client.execute_workflow.await_count == 1


def test_connector_interactive_read_timeout_is_env_configurable(monkeypatch):
    assert _INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS == 120.0
    monkeypatch.setenv(_INTERACTIVE_CONNECTOR_READ_TIMEOUT_ENV.name, '600')
    assert _INTERACTIVE_CONNECTOR_READ_TIMEOUT_ENV.parse_value() == 600.0


@pytest.mark.asyncio
async def test_connector_transport_timeout_stays_sanitized():
    driver, _connectors, _identities, client = _driver(None)
    client.execute_workflow.side_effect = asyncio.TimeoutError(
        'provider-private-transport-timeout')

    with pytest.raises(RpcError) as caught:
        await driver._execute_connector_activity(
            get_connector_sample_rows,
            [],
            timeout=120,
        )

    assert caught.value.message == (
        'Connector operation is temporarily unavailable. Please retry.')
    assert 'provider-private' not in str(caught.value)
    assert 'INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS' not in str(caught.value)


@pytest.mark.asyncio
async def test_connector_execution_result_wait_has_local_deadline():
    driver, _connectors, _identities, client = _driver(None)
    stalled = asyncio.Event()
    client.execute_workflow = MagicMock(return_value=stalled.wait())

    with patch(
            'rest.driver.connector_api_driver.'
            '_CONNECTOR_RESULT_DELIVERY_GRACE_SECONDS', 0):
        with pytest.raises(RpcError) as caught:
            await driver._execute_connector_activity(
                get_connector_sample_rows,
                [],
                timeout=0.01,
            )

    assert caught.value.message == (
        'Connector operation is temporarily unavailable. Please retry.')


@pytest.mark.asyncio
async def test_connector_response_token_result_wait_has_local_deadline():
    stalled = asyncio.Event()
    handle = SimpleNamespace(
        describe=AsyncMock(return_value=SimpleNamespace(
            status=WorkflowExecutionStatus.COMPLETED)),
        result=MagicMock(return_value=stalled.wait()),
    )
    driver, _connectors, _identities, client = _driver(None)
    client.get_workflow_handle = MagicMock(return_value=handle)

    with patch(
            'rest.driver.connector_api_driver.'
            '_CONNECTOR_RESULT_DELIVERY_GRACE_SECONDS', 0):
        with pytest.raises(RpcError) as caught:
            await driver._execute_connector_activity(
                get_connector_sample_rows,
                [],
                response_token=_activity_response_token(
                    get_connector_sample_rows),
                timeout=0.01,
            )

    assert caught.value.message == (
        'Connector operation is temporarily unavailable. Please retry.')
    handle.result.assert_called_once_with()


@pytest.mark.asyncio
async def test_connector_interactive_read_wrapped_temporal_deadline_is_actionable(
):
    driver, _connectors, _identities, client = _driver(None)
    timeout = TimeoutError(
        'provider-private-temporal-timeout',
        type=TimeoutType.START_TO_CLOSE,
        last_heartbeat_details=('provider-private-heartbeat', ),
    )
    client.execute_workflow.side_effect = WorkflowFailureError(cause=timeout)

    with pytest.raises(ConnectorError) as caught:
        await driver._execute_connector_activity(
            get_connector_sample_rows,
            [],
            timeout=120,
        )

    assert caught.value.non_retryable is True
    assert caught.value.message.startswith(
        'Connector source read exceeded the configured 120-second budget.')
    assert 'INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS' in caught.value.message
    assert 'fronting proxy read timeout' in caught.value.message
    assert 'provider-private' not in str(caught.value)
    assert 'retry' not in caught.value.message.lower()


@pytest.mark.parametrize('timeout_type', list(TimeoutType))
@pytest.mark.asyncio
async def test_connector_interactive_read_activity_timeouts_stay_sanitized(
        timeout_type):
    driver, _connectors, _identities, client = _driver(None)
    timeout = TimeoutError(
        'provider-private-activity-timeout',
        type=timeout_type,
        last_heartbeat_details=('provider-private-heartbeat', ),
    )
    activity_error = ActivityError(
        'activity failed',
        scheduled_event_id=1,
        started_event_id=2,
        identity='worker',
        activity_type=get_connector_sample_rows.__name__,
        activity_id='activity-id',
        retry_state=RetryState.NON_RETRYABLE_FAILURE,
    )
    activity_error.__cause__ = timeout
    client.execute_workflow.side_effect = WorkflowFailureError(
        cause=activity_error)

    with pytest.raises(RpcError) as caught:
        await driver._execute_connector_activity(
            get_connector_sample_rows,
            [],
            timeout=120,
        )

    assert caught.value.message == (
        'Connector operation is temporarily unavailable. Please retry.')
    assert 'provider-private' not in str(caught.value)
    assert 'INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS' not in str(caught.value)


@pytest.mark.asyncio
async def test_connector_activity_propagates_request_cancellation():
    driver, _connectors, _identities, client = _driver(None)
    cancellation = asyncio.CancelledError('request cancelled')
    client.execute_workflow.side_effect = cancellation

    with pytest.raises(asyncio.CancelledError) as caught:
        await driver._execute_connector_activity(
            get_connector_sample_rows,
            [],
            timeout=120,
        )

    assert caught.value is cancellation


@pytest.mark.asyncio
async def test_connector_interactive_reads_use_warehouse_safe_timeout():
    expected = MagicMock()
    driver, connectors, _identities, client = _driver(expected)

    result = await driver.get_project_sample_rows(
        PROJECT_ID,
        CONNECTOR_ID,
        MagicMock(),
        'orders',
        num_rows=1,
    )

    assert result is expected
    connectors.execution_ref_for_project.assert_awaited_once()
    call = client.execute_workflow.await_args
    assert call.kwargs['execution_timeout'] == timedelta(
        seconds=_INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS)
    assert call.kwargs['args'] == [
        get_connector_sample_rows.__name__,
        [REFERENCE, 'orders', 1, False, True, True],
    ]

    client.execute_workflow.reset_mock()
    result = await driver.get_project_source_columns(
        PROJECT_ID,
        CONNECTOR_ID,
        MagicMock(),
        'orders',
    )

    assert result is expected
    call = client.execute_workflow.await_args
    assert call.kwargs['execution_timeout'] == timedelta(
        seconds=_INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS)
    assert call.kwargs['args'] == [
        get_connector_source_columns.__name__,
        [REFERENCE, 'orders', True],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'config',
    [
        api.S3ConnectorResourceConfig(target=api.S3AnyURIConnectorTarget()),
        api.GCSConnectorResourceConfig(target=api.GCSAnyURIConnectorTarget()),
        api.ADLSConnectorResourceConfig(target=api.ADLSAnyURIConnectorTarget()),
    ],
    ids=['s3', 'gcs', 'adls'],
)
async def test_any_uri_source_table_listing_rejected_before_activity(config):
    driver, connectors, _identities, client = _driver(None)
    connectors.source_table_connector_for_project.return_value = (
        SimpleNamespace(config=config))

    with pytest.raises(Exception, match='do not support listing'):
        await driver.get_project_source_tables(PROJECT_ID, CONNECTOR_ID,
                                               MagicMock(), sync=True)

    connectors.execution_ref_for_project.assert_not_awaited()
    client.execute_workflow.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('config', 'source'),
    [
        (
            api.S3ConnectorResourceConfig(target=api.S3RootedConnectorTarget(
                bucket='bucket', prefix='root')),
            's3://bucket/root/orders.parquet',
        ),
        (
            api.S3ConnectorResourceConfig(target=api.S3AnyURIConnectorTarget()),
            'orders.parquet',
        ),
        (
            api.GCSConnectorResourceConfig(
                target=api.GCSAnyURIConnectorTarget()),
            'orders.parquet',
        ),
        (
            api.ADLSConnectorResourceConfig(
                target=api.ADLSAnyURIConnectorTarget()),
            'orders.parquet',
        ),
        (
            api.GCSConnectorResourceConfig(target=api.GCSRootedConnectorTarget(
                bucket='bucket', prefix='root')),
            'gs://other-bucket/root/orders.parquet',
        ),
        (
            api.GCSConnectorResourceConfig(target=api.GCSRootedConnectorTarget(
                bucket='bucket', prefix='root')),
            'gs://bucket/root-adjacent/orders.parquet',
        ),
        (
            api.ADLSConnectorResourceConfig(
                target=api.ADLSRootedConnectorTarget(
                    storage_account='account',
                    filesystem='filesystem',
                    prefix='root',
                )),
            ('abfss://filesystem@other.dfs.core.windows.net/'
             'root/orders.parquet'),
        ),
        (
            api.ADLSConnectorResourceConfig(
                target=api.ADLSRootedConnectorTarget(
                    storage_account='account',
                    filesystem='filesystem',
                    prefix='root',
                )),
            ('abfss://other@account.dfs.core.windows.net/'
             'root/orders.parquet'),
        ),
        (
            api.ADLSConnectorResourceConfig(
                target=api.ADLSRootedConnectorTarget(
                    storage_account='account',
                    filesystem='filesystem',
                    prefix='root',
                )),
            ('abfss://filesystem@account.dfs.core.usgovcloudapi.net/'
             'root/orders.parquet'),
        ),
        (
            api.ADLSConnectorResourceConfig(
                target=api.ADLSRootedConnectorTarget(
                    storage_account='account',
                    filesystem='filesystem',
                    prefix='root',
                )),
            ('abfss://filesystem@account.dfs.core.windows.net/'
             'root-adjacent/orders.parquet'),
        ),
    ],
    ids=[
        's3-rooted',
        's3-any-uri',
        'gcs-any-uri',
        'adls-any-uri',
        'gcs-rooted-wrong-bucket',
        'gcs-rooted-wrong-prefix',
        'adls-rooted-wrong-account',
        'adls-rooted-wrong-filesystem',
        'adls-rooted-wrong-endpoint',
        'adls-rooted-wrong-prefix',
    ],
)
async def test_wrong_object_source_mode_rejected_before_activity(
        config, source):
    driver, connectors, _identities, client = _driver(None)
    connectors.source_table_connector_for_project.return_value = (
        SimpleNamespace(config=config))

    with pytest.raises(ValueError):
        await driver.get_project_source_table_file_type(
            PROJECT_ID,
            CONNECTOR_ID,
            MagicMock(),
            source,
        )

    connectors.execution_ref_for_project.assert_not_awaited()
    client.execute_workflow.assert_not_awaited()


@pytest.mark.asyncio
async def test_connector_validation_sends_only_execution_reference():
    driver, connectors, _identities, client = _driver(None)

    await driver.validate_project_connector(PROJECT_ID, CONNECTOR_ID,
                                            MagicMock())

    connectors.execution_ref_for_project.assert_awaited_once()
    activity_inputs = client.execute_workflow.await_args.kwargs['args'][1]
    assert activity_inputs == [REFERENCE]


@pytest.mark.parametrize(
    'identity_id',
    [IDENTITY_ID, None],
    ids=['explicit-identity', 'automatic-identity'],
)
@pytest.mark.asyncio
async def test_connector_create_authorizes_then_sends_only_safe_configuration(
        identity_id):
    expected = MagicMock()
    driver, connectors, _identities, client = _driver(expected)
    args = _databricks_create_args(identity_id)
    principal = MagicMock()

    result = await driver.create_project_connector(PROJECT_ID, args, principal)

    assert result is expected
    connectors.authorize_create.assert_awaited_once_with(
        PROJECT_ID, args, principal)
    call = client.execute_workflow.await_args
    assert call.kwargs['args'][0] == create_project_connector_activity.__name__
    assert call.kwargs['args'][1] == [PROJECT_ID, args, CREATE_AUTHORIZATIONS]
    assert args.identity_id == (str(identity_id)
                                if identity_id is not None else None)
    assert not hasattr(args, 'credentials')
    serialized = repr(call.kwargs['args'][1]).casefold()
    assert 'credential' not in serialized
    assert 'secret' not in serialized


@pytest.mark.parametrize(
    'public_error',
    [
        NoIdentityValidatesTargetError(
            message='No available s3 identity could access this target.'),
        MultipleIdentitiesValidateTargetError(
            message=('Multiple identities can access this target: '
                     'Production A (identity_aaaaaaaa).')),
        IdentityResolutionUnavailableError(
            message=('Connector identity resolution is temporarily '
                     'unavailable; please retry.')),
        ConnectorValidationError(message='Connector target validation failed'),
        ConnectorValidationError(
            message=('Connector target validation failed '
                     '[databricks_sample_table_permissions]; '
                     'reference: 0123456789abcdef0123456789abcdef')),
        IdentityValidationError(message='Identity validation failed'),
    ],
    ids=[
        'no-identity',
        'multiple-identities',
        'identity-resolution-unavailable',
        'connector-validation',
        'connector-validation-reference',
        'identity-validation',
    ],
)
@pytest.mark.asyncio
async def test_connector_resolution_public_status_survives_temporal_wrapping(
        public_error):
    driver, _connectors, _identities, client = _driver(None)
    client.execute_workflow.side_effect = ApplicationError(
        'activity failed',
        public_error.to_dict(),
        type=type(public_error).__name__,
    )
    args = _databricks_create_args(None)

    with pytest.raises(type(public_error)) as caught:
        await driver.create_project_connector(PROJECT_ID, args, MagicMock())

    assert caught.value.http_code == public_error.http_code
    assert str(caught.value) == str(public_error)


@pytest.mark.asyncio
async def test_connector_name_conflict_survives_temporal_wrapping_as_409():
    private_details = (
        'credential=password-secret',
        'provider=databricks',
        'internal=unique-constraint-name',
    )
    private_payload = ' '.join(private_details)
    public_error = ConnectorNameConflictError(
        "Connector named 'warehouse' already exists in this project")
    driver, _connectors, _identities, client = _driver(None)
    client.execute_workflow.side_effect = ApplicationError(
        f'activity failed: {private_payload}',
        public_error.to_dict(),
        type=type(public_error).__name__,
        non_retryable=public_error.non_retryable,
    )

    with pytest.raises(ConnectorNameConflictError) as caught:
        await driver.create_project_connector(PROJECT_ID,
                                              _databricks_create_args(),
                                              MagicMock())

    reconstructed = caught.value
    assert type(reconstructed) is ConnectorNameConflictError
    assert isinstance(reconstructed, ValueError)
    assert reconstructed.http_code == 409
    assert reconstructed.non_retryable is True
    assert str(reconstructed) == str(public_error)
    assert all(detail not in str(reconstructed) for detail in private_details)
    assert reconstructed.__cause__ is None
    assert reconstructed.__context__ is None


@pytest.mark.parametrize('activity_fn', [
    get_connector_source_columns,
    get_connector_sample_rows,
])
@pytest.mark.asyncio
async def test_missing_table_survives_temporal_wrapping_as_404(activity_fn):
    public_error = ConnectorSourceTableNotFoundError(
        'Unable to find table orders')
    driver, _connectors, _identities, client = _driver(None)
    client.execute_workflow.side_effect = ApplicationError(
        'activity failed',
        public_error.to_dict(),
        type=type(public_error).__name__,
    )

    with pytest.raises(ConnectorSourceTableNotFoundError) as caught:
        await driver._execute_connector_activity(activity_fn, [])

    assert type(caught.value) is ConnectorSourceTableNotFoundError
    assert caught.value.http_code == 404
    assert str(caught.value) == 'Unable to find table orders'


@pytest.mark.asyncio
async def test_connector_activity_rejects_direct_same_name_public_impostor():
    private_details = (
        'credential=direct-impostor-secret',
        'provider=direct-impostor-provider',
        'internal=direct-impostor-detail',
    )
    impostor_type = type(
        'ConnectorNameConflictError',
        (ConnectorNameConflictError, ),
        {},
    )
    impostor = impostor_type(message=' '.join(private_details))
    assert type(impostor) is not ConnectorNameConflictError
    assert isinstance(impostor, ConnectorNameConflictError)
    driver, _connectors, _identities, client = _driver(None)
    client.execute_workflow.side_effect = impostor

    try:
        with pytest.raises(RpcError) as caught:
            await driver._execute_connector_activity(
                create_project_connector_activity,
                [],
            )

        for detail in private_details:
            _assert_sanitized_internal_error(caught.value, detail)
        assert caught.value.message == (
            'Connector operation is temporarily unavailable. Please retry.')
        assert caught.value.stack_trace == ''
    finally:
        client.execute_workflow.side_effect = None
        del impostor
        del impostor_type
        gc.collect()


@pytest.mark.asyncio
async def test_connector_activity_rejects_same_name_temporal_impostor():
    private_details = (
        'credential=temporal-impostor-secret',
        'provider=temporal-impostor-provider',
        'internal=temporal-impostor-detail',
    )
    impostor_type = type(
        'ConnectorNameConflictError',
        (ResourceSlugConflictError, ),
        {},
    )
    impostor = impostor_type(message=' '.join(private_details))
    assert type(impostor) is not ConnectorNameConflictError
    payload = impostor.to_dict()
    assert type(PublicException.from_dict(payload)) is impostor_type
    driver, _connectors, _identities, client = _driver(None)
    client.execute_workflow.side_effect = ApplicationError(
        'activity failed',
        payload,
        type='ConnectorNameConflictError',
        non_retryable=impostor.non_retryable,
    )

    try:
        with pytest.raises(RpcError) as caught:
            await driver._execute_connector_activity(
                create_project_connector_activity,
                [],
            )

        for detail in private_details:
            _assert_sanitized_internal_error(caught.value, detail)
        assert caught.value.message == (
            'Connector operation is temporarily unavailable. Please retry.')
        assert caught.value.stack_trace == ''
    finally:
        client.execute_workflow.side_effect = None
        del impostor
        del impostor_type
        gc.collect()


@pytest.mark.parametrize(
    ('activity_fn', 'response_token'),
    [
        (validate_identity_credentials_activity, ''),
        (
            create_project_connector_activity,
            _activity_response_token(validate_identity_credentials_activity),
        ),
    ],
    ids=['alternate-activity', 'mismatched-response-token'],
)
@pytest.mark.asyncio
async def test_connector_activity_rejects_cross_process_conflict_from_untrusted_producer(
        activity_fn, response_token):
    private_details = (
        'credential=cross-process-secret',
        'provider=cross-process-provider',
        'internal=cross-process-detail',
    )
    payload = _isolated_name_conflict_payload(' '.join(private_details))
    reconstructed = PublicException.from_dict(payload)
    assert type(reconstructed) is ConnectorNameConflictError
    failure = ApplicationError(
        'activity failed',
        payload,
        type=ConnectorNameConflictError.__name__,
    )
    driver, _connectors, _identities, client = _driver(None)
    kwargs = {}
    if response_token:
        handle = SimpleNamespace(
            describe=AsyncMock(return_value=SimpleNamespace(
                status=WorkflowExecutionStatus.COMPLETED)),
            result=AsyncMock(side_effect=failure),
        )
        client.get_workflow_handle = MagicMock(return_value=handle)
        kwargs['response_token'] = response_token
    else:
        client.execute_workflow.side_effect = failure

    with pytest.raises(RpcError) as caught:
        await driver._execute_connector_activity(
            activity_fn,
            [],
            **kwargs,
        )

    for detail in private_details:
        _assert_sanitized_internal_error(caught.value, detail)
    assert caught.value.message == (
        'Connector operation is temporarily unavailable. Please retry.')
    assert caught.value.stack_trace == ''


@pytest.mark.parametrize(
    'failure',
    [
        RuntimeError('transport outage provider-signed-request-secret'),
        ApplicationError(
            'worker unavailable provider-signed-request-secret',
            type='NotFoundError',
        ),
        ApplicationError(
            'value error provider-signed-request-secret',
            type='ValueError',
        ),
        TimeoutError(
            'timeout provider-signed-request-secret',
            type=TimeoutType.START_TO_CLOSE,
            last_heartbeat_details=('provider-signed-request-secret', ),
        ),
    ],
    ids=[
        'transport-outage',
        'worker-unavailable',
        'generic-value-error',
        'timeout',
    ],
)
@pytest.mark.asyncio
async def test_connector_activity_redacts_unknown_system_failure(failure):
    secret = 'provider-signed-request-secret'
    driver, _connectors, _identities, client = _driver(None)
    client.execute_workflow.side_effect = failure
    args = api.CreateConnectorArgs(
        name='warehouse',
        config=api.S3ConnectorResourceConfig(
            target=api.S3RootedConnectorTarget(
                bucket='company-data',
                prefix='orders',
                region='us-west-2',
            ), ),
    )

    with pytest.raises(RpcError) as caught:
        await driver.create_project_connector(PROJECT_ID, args, MagicMock())

    _assert_sanitized_internal_error(caught.value, secret)
    assert caught.value.message == (
        'Connector operation is temporarily unavailable. Please retry.')
    assert caught.value.stack_trace == ''


@pytest.mark.parametrize(
    'public_error_type',
    [ConnectorValidationError, ConnectorNameConflictError],
    ids=['validation', 'name-conflict'],
)
@pytest.mark.asyncio
async def test_connector_activity_rejects_forged_allowlisted_envelope(
        public_error_type):
    secret = 'forged-public-provider-secret'
    forged = public_error_type(message=secret).to_dict()
    forged['http_code'] = 503
    driver, _connectors, _identities, client = _driver(None)
    client.execute_workflow.side_effect = ApplicationError(
        'activity failed',
        forged,
        type=public_error_type.__name__,
    )

    with pytest.raises(RpcError) as caught:
        await driver._execute_connector_activity(
            create_project_connector_activity,
            [],
        )

    _assert_sanitized_internal_error(caught.value, secret)
    assert caught.value.message == (
        'Connector operation is temporarily unavailable. Please retry.')
    assert caught.value.stack_trace == ''


@pytest.mark.parametrize(
    'stage',
    [
        'client-setup',
        'response-token-describe',
        'response-token-result',
        'start-workflow',
    ],
)
@pytest.mark.asyncio
async def test_connector_activity_redacts_failures_across_all_client_stages(
        stage):
    secret = 'temporal-client-stage-secret'
    driver, _connectors, _identities, client = _driver(None)
    kwargs = {}
    if stage == 'client-setup':
        driver._data_backend_ctx.temporal_client().get.side_effect = (
            RuntimeError(f'client setup failed: {secret}'))
    elif stage.startswith('response-token'):
        if stage == 'response-token-describe':
            handle = SimpleNamespace(
                describe=AsyncMock(
                    side_effect=RuntimeError(f'describe failed: {secret}')), )
        else:
            handle = SimpleNamespace(
                describe=AsyncMock(return_value=SimpleNamespace(
                    status=WorkflowExecutionStatus.COMPLETED)),
                result=AsyncMock(
                    side_effect=RuntimeError(f'result failed: {secret}')),
            )
        client.get_workflow_handle = MagicMock(return_value=handle)
        kwargs['response_token'] = 'safe-response-token'
    else:
        client.start_workflow = AsyncMock(
            side_effect=RuntimeError(f'start failed: {secret}'))
        kwargs['asynchronous'] = True

    with pytest.raises(RpcError) as caught:
        await driver._execute_connector_activity(
            create_project_connector_activity,
            [],
            **kwargs,
        )

    _assert_sanitized_internal_error(caught.value, secret)


@pytest.mark.asyncio
async def test_connector_response_token_preserves_allowlisted_completion_error(
):
    public_error = NoIdentityValidatesTargetError(
        message='No available s3 identity could access this target.')
    handle = SimpleNamespace(
        describe=AsyncMock(return_value=SimpleNamespace(
            status=WorkflowExecutionStatus.COMPLETED)),
        result=AsyncMock(side_effect=ApplicationError(
            'activity failed',
            public_error.to_dict(),
            type=type(public_error).__name__,
        )),
    )
    driver, _connectors, _identities, client = _driver(None)
    client.get_workflow_handle = MagicMock(return_value=handle)

    with pytest.raises(NoIdentityValidatesTargetError) as caught:
        await driver._execute_connector_activity(
            create_project_connector_activity,
            [],
            response_token='safe-response-token',
        )

    assert str(caught.value) == str(public_error)
    assert caught.value.http_code == 422


@pytest.mark.asyncio
async def test_connector_response_token_preserves_authenticated_name_conflict():
    public_error = ConnectorNameConflictError(
        "Connector named 'warehouse' already exists in this project")
    handle = SimpleNamespace(
        describe=AsyncMock(return_value=SimpleNamespace(
            status=WorkflowExecutionStatus.COMPLETED)),
        result=AsyncMock(side_effect=ApplicationError(
            'activity failed',
            public_error.to_dict(),
            type=type(public_error).__name__,
        )),
    )
    driver, _connectors, _identities, client = _driver(None)
    client.get_workflow_handle = MagicMock(return_value=handle)

    with pytest.raises(ConnectorNameConflictError) as caught:
        await driver._execute_connector_activity(
            create_project_connector_activity,
            [],
            response_token=_activity_response_token(
                create_project_connector_activity),
        )

    assert type(caught.value) is ConnectorNameConflictError
    assert str(caught.value) == str(public_error)
    assert caught.value.http_code == 409


@pytest.mark.asyncio
async def test_identity_test_authorizes_then_sends_only_identity_id():
    expected = MagicMock()
    driver, _connectors, identities, client = _driver(expected)
    principal = MagicMock()

    result = await driver.test_identity_credentials(IDENTITY_ID, principal)

    assert result is expected
    identities.authorize_test_credentials.assert_awaited_once_with(
        IDENTITY_ID, principal)
    call = client.execute_workflow.await_args
    assert call.kwargs['args'][
        0] == validate_identity_credentials_activity.__name__
    assert call.kwargs['args'][1] == [IDENTITY_ID, AUTHORIZATION]


@pytest.mark.parametrize(
    'identity_id',
    [IDENTITY_ID, None],
    ids=['explicit-identity', 'automatic-identity'],
)
def test_connector_create_activity_uses_activity_only_service_path(identity_id):
    expected = MagicMock()
    authorization = SimpleNamespace(
        principal_id='submitter',
        auth_backend='direct_connect',
    )
    service = SimpleNamespace(create_from_activity=AsyncMock(
        return_value=expected))
    args = _databricks_create_args(identity_id)

    result, broker = _run_activity_service(
        SimpleNamespace(connectors=service),
        authorization,
        'authorize_project_activity_scope',
        lambda: create_project_connector_activity(PROJECT_ID, args,
                                                  CREATE_AUTHORIZATIONS),
    )

    assert result is expected
    assert args.identity_id == (str(identity_id)
                                if identity_id is not None else None)
    broker.assert_called_once()
    service.create_from_activity.assert_awaited_once_with(
        PROJECT_ID, args, UserID('submitter'), {IDENTITY_ID})


def test_connector_create_activity_preserves_exact_name_conflict():
    public_error = ConnectorNameConflictError(
        "Connector named 'warehouse' already exists in this project")
    authorization = SimpleNamespace(
        principal_id='submitter',
        auth_backend='direct_connect',
    )
    service = SimpleNamespace(create_from_activity=AsyncMock(
        side_effect=public_error))

    with pytest.raises(ConnectorNameConflictError) as caught:
        _run_activity_service(
            SimpleNamespace(connectors=service),
            authorization,
            'authorize_project_activity_scope',
            lambda: create_project_connector_activity(
                PROJECT_ID, _databricks_create_args(), CREATE_AUTHORIZATIONS),
        )

    assert caught.value is public_error
    assert type(caught.value) is ConnectorNameConflictError
    assert isinstance(caught.value, ValueError)
    assert caught.value.http_code == 409


@pytest.mark.parametrize(
    ('impostor_name', 'impostor_base'),
    [
        ('ConnectorNameConflictError', ConnectorNameConflictError),
        ('PrivateConnectorNameConflictError', ConnectorNameConflictError),
        ('ConnectorNameConflictError', ResourceSlugConflictError),
    ],
    ids=['same-name-subclass', 'renamed-subclass', 'same-name-unrelated'],
)
def test_connector_create_activity_rejects_nonexact_name_conflict(
        impostor_name, impostor_base):
    secret = 'credential=producer-impostor-secret'
    impostor_type = type(
        impostor_name,
        (impostor_base, ),
        {},
    )
    impostor = impostor_type(message=secret)
    if impostor_name == ConnectorNameConflictError.__name__:
        payload = impostor.to_dict()
        reconstructed = PublicException.from_dict(payload)
        assert type(
            reconstructed).__name__ == ConnectorNameConflictError.__name__
    authorization = SimpleNamespace(
        principal_id='submitter',
        auth_backend='direct_connect',
    )
    service = SimpleNamespace(create_from_activity=AsyncMock(
        side_effect=impostor))

    try:
        with pytest.raises(ValueError) as caught:
            _run_activity_service(
                SimpleNamespace(connectors=service),
                authorization,
                'authorize_project_activity_scope',
                lambda: create_project_connector_activity(
                    PROJECT_ID, _databricks_create_args(), CREATE_AUTHORIZATIONS
                ),
            )

        assert type(caught.value) is ValueError
        assert secret not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
    finally:
        service.create_from_activity.side_effect = None
        del impostor
        del impostor_type
        gc.collect()


def test_identity_test_activity_uses_activity_only_service_path():
    expected = MagicMock()
    principal = MagicMock()
    service = SimpleNamespace(test_credentials_from_activity=AsyncMock(
        return_value=expected))
    result, _ = _run_activity_service(
        SimpleNamespace(identities=service),
        principal,
        'authorize_activity_scope',
        lambda: validate_identity_credentials_activity(IDENTITY_ID,
                                                       AUTHORIZATION),
    )

    assert result is expected
    service.test_credentials_from_activity.assert_awaited_once_with(
        IDENTITY_ID, principal)


def test_source_table_activity_resolves_and_closes_in_activity_process():
    connector = MagicMock()
    connector.list_tables.return_value = ['orders']
    resolved = _resolved_external_spec()
    with patch(_BROKER_RESOLVE, return_value=resolved) as broker, patch(
            _CONNECTOR_FACTORY, return_value=connector) as resolve:
        response = list_connector_source_tables(REFERENCE, [])

    broker.assert_called_once_with(REFERENCE)
    resolve.assert_called_once_with(resolved)
    connector.close.assert_called_once_with()
    assert [table.name for table in response.source_tables] == ['orders']


@pytest.mark.parametrize('rootless', [False, True])
def test_source_download_is_staged_inside_connector_scope(rootless):
    from kumoapi.s3 import normalize_s3_source_uri

    from kumo.connector.any_uri_s3_connector import AnyURIS3Connector

    connector = MagicMock(
        spec=AnyURIS3Connector if rootless else CompositeFileConnector)
    connector._normalizer = normalize_s3_source_uri
    source = 's3://customer/table/' if rootless else 'table/'
    connector.filesystem_credential_scope.return_value = nullcontext()
    connector._get_path.return_value = 's3://customer/table.parquet'
    connector._supported_file_type.return_value = 'parquet'
    connector._local_source.return_value = nullcontext(
        (SimpleNamespace(directory='/local/prepared'), 'parquet'))
    with patch(_CONNECTOR_FACTORY, return_value=connector), patch(
            'rest.driver.connector_api_driver.deploy_env',
            return_value=SimpleNamespace(
                data_plane_storage_root='s3://example-internal')), patch(
                    'rest.driver.connector_api_driver.fsutil') as filesystem:
        filesystem.isdir.return_value = False
        result = stage_connector_source_download(REFERENCE, source,
                                                 'staging123')

    assert result.endswith('/downloads/staging123/table')
    filesystem.copy.assert_called_once()
    connector._local_source.assert_called_once_with(source, transform=False,
                                                    max_entries=200,
                                                    max_bytes=1 * 10**9)
    connector.close.assert_called_once_with()


@pytest.mark.parametrize('source', [
    '../other/table.parquet',
    '/other/table.parquet',
    's3://other/table.parquet',
    'table/../../other.parquet',
    'table/%2e%2e/other.parquet',
])
def test_source_download_rejects_root_escape_before_acquisition(source):
    connector = MagicMock(spec=CompositeFileConnector)
    with patch(_CONNECTOR_FACTORY, return_value=connector):
        with pytest.raises(ConnectorValidationError):
            stage_connector_source_download(REFERENCE, source, 'staging123')

    connector.filesystem_credential_scope.assert_not_called()
    connector._get_path.assert_not_called()
    connector._local_source.assert_not_called()
    connector.close.assert_called_once_with()


def test_fixed_schema_databricks_activity_discovers_bare_table_names():
    import pyarrow as pa

    connector = DatabricksConnector(
        name='databricks',
        host='https://workspace.cloud.databricks.com',
        cluster_id='',
        warehouse_id='warehouse-id',
        catalog='main',
        schema_name='sales',
    )
    connector._context = MagicMock()
    cursor = connector.context.cursor.return_value.__enter__.return_value
    cursor.columns.return_value.fetchall_arrow.return_value = pa.table({
        'TABLE_SCHEM': ['sales', 'other'],
        'TABLE_NAME': ['orders', 'private_orders'],
        'COLUMN_NAME': ['id', 'id'],
        'TYPE_NAME': ['BIGINT', 'BIGINT'],
    })

    response = _get_source_tables_raw(
        CONNECTOR_ID,
        connector,
        [],
        _resolved_external_spec(),
    )

    assert [table.name for table in response.source_tables] == ['orders']
    cursor.columns.assert_called_once_with(catalog_name='main',
                                           schema_name='sales')


@pytest.mark.parametrize(
    ('expected', 'coordinate'),
    [
        (True, 's3://customer/orders.parquet'),
        (False, 'abfss://data@account.dfs.core.windows.net/orders.parquet'),
    ],
    ids=['exists', 'missing'],
)
def test_exact_source_validation_bypasses_cache_and_closes(
        expected, coordinate):
    connector = MagicMock()
    connector.table_exists.return_value = expected

    with patch(_CONNECTOR_FACTORY, return_value=connector) as resolve:
        result = validate_connector_source_table_exists(REFERENCE, coordinate)

    assert result is expected
    resolve.assert_called_once_with(REFERENCE)
    connector.table_exists.assert_called_once_with(coordinate)
    connector.list_tables.assert_not_called()
    connector.close.assert_called_once_with()


def test_source_type_resolution_preserves_the_validation_message():
    connector = MagicMock(spec=CompositeFileConnector)
    connector.get_path_and_type.side_effect = ValidationException(
        title='Bad Path', message='Select a table below it, such as: x/dims.')

    with patch(_CONNECTOR_FACTORY, return_value=connector):
        with pytest.raises(ValidationException, match='x/dims'):
            get_connector_source_table_file_type(REFERENCE, 'x')


def test_external_file_discovery_never_performs_unscoped_root_probe(
        monkeypatch):
    connector = MagicMock(spec=FileConnector)
    connector.list_tables.return_value = ['deep/orders.parquet']
    ambient_probe = MagicMock(
        side_effect=AssertionError('ambient filesystem provider was touched'))
    monkeypatch.setattr('rest.driver.connector_api_driver.fsutil.listdir',
                        ambient_probe)

    response = _get_source_tables_raw(
        CONNECTOR_ID,
        connector,
        [],
        _resolved_external_spec(),
    )

    ambient_probe.assert_not_called()
    connector.list_tables.assert_called_once_with(None)
    assert [table.name
            for table in response.source_tables] == ['deep/orders.parquet']


def test_only_canonical_project_upload_discovery_handles_empty_root(
        monkeypatch):
    connector = MagicMock(spec=FileConnector)
    connector.root_dir = '/internal/project-upload'
    connector.list_tables.side_effect = AssertionError(
        'empty internal upload root must not be listed twice')
    root_probe = MagicMock(return_value=[])
    monkeypatch.setattr('rest.driver.connector_api_driver.fsutil.listdir',
                        root_probe)

    response = _get_source_tables_raw(
        CONNECTOR_ID,
        connector,
        [],
        _resolved_internal_spec(is_project_upload=True),
    )

    root_probe.assert_called_once_with('/internal/project-upload')
    connector.list_tables.assert_not_called()
    assert response.source_tables == []


def test_forged_legacy_upload_flag_cannot_enable_internal_root_probe(
        monkeypatch):
    connector = MagicMock(spec=FileConnector)
    connector.list_tables.return_value = ['orders.parquet']
    ambient_probe = MagicMock(
        side_effect=AssertionError('ambient filesystem provider was touched'))
    monkeypatch.setattr('rest.driver.connector_api_driver.fsutil.listdir',
                        ambient_probe)
    resolved = _resolved_external_spec()

    with patch(_BROKER_RESOLVE,
               return_value=resolved), patch(_CONNECTOR_FACTORY,
                                             return_value=connector):
        # The former third positional privilege bit is now only a schema
        # argument. Even a truthy value cannot trigger the internal probe.
        response = list_connector_source_tables(REFERENCE, [], True)
        with pytest.raises(TypeError):
            list_connector_source_tables(REFERENCE, [], True, None)

    ambient_probe.assert_not_called()
    connector.list_tables.assert_called_once_with(True)
    assert [table.name
            for table in response.source_tables] == ['orders.parquet']


def test_stale_upload_reference_fails_before_any_filesystem_access(monkeypatch):
    filesystem_access = MagicMock(
        side_effect=AssertionError('filesystem must not be touched'))
    monkeypatch.setattr('rest.driver.connector_api_driver.fsutil.listdir',
                        filesystem_access)
    connector_factory = MagicMock()

    with patch(
            _BROKER_RESOLVE,
            side_effect=PermissionError('connector is no longer active'),
    ), patch(_CONNECTOR_FACTORY, connector_factory):
        with pytest.raises(ConnectorValidationError):
            list_connector_source_tables(REFERENCE, [])

    filesystem_access.assert_not_called()
    connector_factory.assert_not_called()


def test_validation_activity_resolves_before_native_validation():
    resolved = object()
    with patch(_BROKER_RESOLVE, return_value=resolved) as resolve, patch(
            'rest.config.identity.validate_resolved_connector') as validate:
        validate_connector_reference(REFERENCE)

    resolve.assert_called_once_with(REFERENCE)
    validate.assert_called_once_with(resolved)


def test_validation_activity_redacts_provider_exception_text():
    secret = 'provider-signed-assertion-secret'
    with patch(_BROKER_RESOLVE, return_value=object()), patch(
            'rest.config.identity.validate_resolved_connector',
            side_effect=RuntimeError(f'permission denied: {secret}')):
        error = _assert_connector_failure_is_sanitized(
            lambda: validate_connector_reference(REFERENCE), secret)

    # An unclassified provider failure keeps the generic message and gains a
    # correlation id (which ties the response to the logged cause), but no
    # diagnostic category and no provider text.
    assert str(error).startswith('Connector target validation failed')
    assert '; reference: ' in str(error)
    assert '[' not in str(error)


def test_source_table_failure_log_redacts_provider_exception_text(caplog):
    secret = 'provider-bearer-token-secret'
    connector = MagicMock()
    connector.name = 'warehouse'
    connector.list_tables.side_effect = RuntimeError(
        f'request failed with token {secret}')

    _assert_connector_failure_is_sanitized(
        lambda: _get_source_tables_raw(CONNECTOR_ID, connector, [],
                                       _resolved_external_spec()),
        secret,
        caplog,
    )

    assert 'RuntimeError' in caplog.text


@pytest.mark.parametrize('operation', [
    'discovery',
    'columns',
    'sample',
    'view',
])
def test_connector_source_activity_provider_failures_are_sanitized(
        caplog, operation) -> None:
    secret = 'signed-url=https://secret&token=provider-secret'
    connector = MagicMock()
    connector.list_tables.side_effect = RuntimeError(secret)
    connector.is_table_view.side_effect = RuntimeError(secret)

    with (
            patch(_CONNECTOR_FACTORY, return_value=connector),
            patch(_BROKER_RESOLVE, return_value=_resolved_external_spec()),
            patch('rest.driver.connector_api_driver.get_columns_response',
                  side_effect=RuntimeError(secret)),
            patch('rest.driver.connector_api_driver.get_sample_rows_response',
                  side_effect=RuntimeError(secret)),
    ):

        def invoke():
            if operation == 'discovery':
                return list_connector_source_tables(REFERENCE, [])
            elif operation == 'columns':
                return get_connector_source_columns(REFERENCE, 'orders')
            elif operation == 'sample':
                return get_connector_sample_rows(
                    REFERENCE,
                    'orders',
                    5,
                    serialize_to_json=True,
                    skip_cache=True,
                    require_stats=False,
                )
            return is_connector_source_table_view(REFERENCE, 'orders')

        _assert_connector_failure_is_sanitized(invoke, secret, caplog)

    connector.close.assert_called_once_with()


def _read_connector_source(operation):
    if operation == 'columns':
        return get_connector_source_columns(REFERENCE, 'orders')
    return get_connector_sample_rows(
        REFERENCE,
        'orders',
        5,
        serialize_to_json=True,
        skip_cache=True,
        require_stats=False,
    )


@pytest.mark.parametrize('operation', ['columns', 'sample'])
def test_connector_source_activity_missing_table_is_typed(operation) -> None:
    private_path = 's3://company-data/orders/private.parquet'
    connector = MagicMock()

    with (
            patch(_CONNECTOR_FACTORY, return_value=connector),
            patch(_BROKER_RESOLVE, return_value=_resolved_external_spec()),
            patch('rest.driver.connector_api_driver.get_columns_response',
                  side_effect=FileNotFoundError(private_path)),
            patch('rest.driver.connector_api_driver.get_sample_rows_response',
                  side_effect=FileNotFoundError(private_path)),
    ):
        with pytest.raises(ConnectorSourceTableNotFoundError) as caught:
            _read_connector_source(operation)

    assert type(caught.value) is ConnectorSourceTableNotFoundError
    assert str(caught.value) == 'Unable to find table orders'
    assert private_path not in str(caught.value)
    connector.close.assert_called_once_with()


@pytest.mark.parametrize('operation', ['columns', 'sample'])
@pytest.mark.parametrize(
    'error_type', [SourceTableNotFoundError, SourceTableUnavailableError])
def test_connector_source_activity_maps_typed_table_error_to_404(
        operation, error_type) -> None:
    connector = MagicMock()
    source_error = error_type('orders', 'warehouse')

    with (
            patch(_CONNECTOR_FACTORY, return_value=connector),
            patch(_BROKER_RESOLVE, return_value=_resolved_external_spec()),
            patch('rest.driver.connector_api_driver.get_columns_response',
                  side_effect=source_error),
            patch('rest.driver.connector_api_driver.get_sample_rows_response',
                  side_effect=source_error),
            pytest.raises(ConnectorSourceTableNotFoundError) as caught,
    ):
        _read_connector_source(operation)

    assert type(caught.value) is ConnectorSourceTableNotFoundError
    assert caught.value.http_code == 404
    assert str(caught.value) == 'Unable to find table orders'
    connector.close.assert_called_once_with()


@pytest.mark.parametrize(
    'error_type', [SourceTableNotFoundError, SourceTableUnavailableError])
def test_exact_source_validation_maps_typed_table_error_to_false(
        error_type) -> None:
    connector = MagicMock()
    connector.table_exists.side_effect = error_type('orders', 'warehouse')

    with patch(_CONNECTOR_FACTORY, return_value=connector):
        result = validate_connector_source_table_exists(REFERENCE, 'orders')

    assert result is False
    connector.close.assert_called_once_with()


@pytest.mark.parametrize('operation', ['columns', 'sample'])
@pytest.mark.parametrize('impostor_base', [
    ConnectorValidationError,
    ValidationException,
])
def test_connector_source_activity_sanitizes_not_found_impostor(
        operation, impostor_base) -> None:
    secret = 'credential=not-found-impostor-secret'
    impostor_type = type(
        ConnectorSourceTableNotFoundError.__name__,
        (impostor_base, ),
        {},
    )
    impostor = impostor_type(message=secret)
    connector = MagicMock()

    try:
        with (
                patch(_CONNECTOR_FACTORY, return_value=connector),
                patch(_BROKER_RESOLVE, return_value=_resolved_external_spec()),
                patch('rest.driver.connector_api_driver.get_columns_response',
                      side_effect=impostor),
                patch(
                    'rest.driver.connector_api_driver.'
                    'get_sample_rows_response', side_effect=impostor),
        ):
            with pytest.raises(ConnectorValidationError) as caught:
                _read_connector_source(operation)

        assert type(caught.value) is ConnectorValidationError
        assert str(caught.value) == 'Connector source access failed'
        assert secret not in str(caught.value)
        assert caught.value.__cause__ is None
    finally:
        caught = None
        del impostor
        del impostor_type
        gc.collect()


@pytest.mark.parametrize('failure_stage', ['resolve', 'close'])
def test_connector_source_activity_resolution_and_close_are_sanitized(
        caplog, failure_stage) -> None:
    secret = 'provider-account-secret-token'
    connector = MagicMock()
    connector.list_tables.return_value = []
    resolved = _resolved_external_spec()
    if failure_stage == 'close':
        connector.close.side_effect = RuntimeError(secret)
        resolver = MagicMock(return_value=connector)
        broker = MagicMock(return_value=resolved)
    else:
        resolver = MagicMock(return_value=connector)
        broker = MagicMock(side_effect=RuntimeError(secret))

    with patch(_CONNECTOR_FACTORY, resolver), patch(_BROKER_RESOLVE, broker):
        _assert_connector_failure_is_sanitized(
            lambda: list_connector_source_tables(REFERENCE, []),
            secret,
            caplog,
        )


@pytest.mark.parametrize('operation', ['columns', 'sample'])
def test_query_path_warehouse_error_propagates_unflattened(operation) -> None:
    """A ``CustomSqlQueryError`` from the Custom SQL query path is on the
    boundary allowlist, so its actionable warehouse message reaches the client
    verbatim instead of being flattened to "Connector source access failed"."""
    connector = MagicMock()
    query_error = CustomSqlQueryError(
        'The warehouse could not run this query. '
        'SQL compilation error: Unknown function AI_GEN')

    with (
            patch('rest.driver.connector_api_driver.resolve_connector_and_type',
                  return_value=(connector, ConnectorType.SNOWFLAKE)),
            patch('rest.driver.connector_api_driver.get_query_columns_response',
                  side_effect=query_error),
            patch(
                'rest.driver.connector_api_driver.'
                'get_query_sample_rows_response', side_effect=query_error),
    ):
        with pytest.raises(CustomSqlQueryError) as caught:
            if operation == 'columns':
                get_connector_source_columns(REFERENCE, '',
                                             query="SELECT ai_gen('x') AS ai")
            else:
                get_connector_sample_rows(REFERENCE, '', 5,
                                          serialize_to_json=True,
                                          skip_cache=True, require_stats=False,
                                          query="SELECT ai_gen('x') AS ai")

    message = str(caught.value)
    assert 'Unknown function AI_GEN' in message
    assert 'source access' not in message.lower()
    connector.close.assert_called_once_with()


@pytest.mark.parametrize('operation', ['columns', 'sample'])
def test_query_path_connection_failure_stays_source_access(caplog,
                                                           operation) -> None:
    """A raw (unwrapped) failure on the query path — e.g. a connection/auth
    error the query helper never re-classified — still flattens to the generic
    "Connector source access failed", with the provider text redacted."""
    secret = 'account-url=https://secret&token=provider-secret'
    connector = MagicMock()

    with (
            patch('rest.driver.connector_api_driver.resolve_connector_and_type',
                  return_value=(connector, ConnectorType.SNOWFLAKE)),
            patch('rest.driver.connector_api_driver.get_query_columns_response',
                  side_effect=RuntimeError(secret)),
            patch(
                'rest.driver.connector_api_driver.'
                'get_query_sample_rows_response',
                side_effect=RuntimeError(secret)),
    ):
        with pytest.raises(ConnectorValidationError) as caught:
            if operation == 'columns':
                get_connector_source_columns(REFERENCE, '', query='SELECT 1')
            else:
                get_connector_sample_rows(REFERENCE, '', 5,
                                          serialize_to_json=True,
                                          skip_cache=True, require_stats=False,
                                          query='SELECT 1')

    assert str(caught.value) == 'Connector source access failed'
    assert not isinstance(caught.value, CustomSqlQueryError)
    assert secret not in str(caught.value)
    assert secret not in caplog.text
    assert caught.value.__cause__ is None
    connector.close.assert_called_once_with()
