# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import os.path as osp
import uuid
from datetime import timedelta
from typing import Dict, List, Optional

import humanize
import kumoapi.data_source as api
import kumoapi.source_table as source_api
from kumo_core.types import ConnectorType
from kumo_databricks.sql import normalize_table_coordinate
from kumoapi.s3 import (
    normalize_s3_source_relative_path,
    normalize_s3_source_uri,
)
from temporalio.client import WorkflowExecutionStatus, WorkflowFailureError
from temporalio.exceptions import TimeoutError as TemporalTimeoutError
from temporalio.exceptions import TimeoutType

from async_utils.sync import FutureDedupeOption, blocking_func, blocking_method
from common_utils.string import validate_connector_relative_source_name
from io_utils import fsutil, logging
from io_utils.fsutil.impl_local import (
    local_upload_part_path,
    validate_local_upload_path,
)
from kumo.connector.any_uri_file_connector import AnyURIFileConnector
from kumo.connector.any_uri_s3_connector import AnyURIS3Connector
from kumo.connector.bigquery_connector import BigQueryConnector
from kumo.connector.composite_file_connector import CompositeFileConnector
from kumo.connector.connector import (
    Connector,
    ConnectorError,
    SourceTableNotFoundError,
    SourceTableUnavailableError,
)
from kumo.connector.databricks_connector import DatabricksConnector
from kumo.connector.file_connector import (
    MAX_SUPPORTED_FILES,
    FileConnector,
    _list_dataset_files,
)
from kumo.connector.snowflake_connector import SnowflakeConnector
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from kumo.validation import ValidationException
from rest.backends.data_backend import connector_activity_registry
from rest.backends.data_backend import data_backend as data_backend_module
from rest.backends.data_backend.activities.identity_management import (
    create_project_connector_activity,
    validate_identity_credentials_activity,
)
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.utils import resolve_connector_and_type
from rest.config.connector import (
    FileConnectorResourceConfig,
    SampleRowsResponse,
    SourceColumnsResource,
    SourceTable,
    SourceTablesResponse,
    object_store_source_for_connector,
)
from rest.config.data_backend import (
    ConnectorExecutionRef,
    ConnectorOpenSpec,
    ConnectorSpec,
)
from rest.config.table_v2 import DataSourceType, TableDBEntryV2
from rest.constants import DATA_BACKEND_TASK_QUEUE
from rest.driver.error_constants import (
    FILE_SIZE_EXCEEDED_MESSAGE,
    FILE_UPLOAD_NOT_SUPPORTED_MESSAGE,
)
from rest.settings import DataScaleLimits
from rest.utils.connector import (
    DEFAULT_NUM_SAMPLE_ROWS,
    get_columns_response,
    get_query_columns_response,
    get_query_sample_rows_response,
    get_sample_rows_response,
)
from rest.utils.driver_common import get_connector_from_spec
from rest.utils.file_upload import supports_backend_file_upload
from rest.utils.public import presign_parquet_table_data
from rest.utils.table import validate_csv_file, validate_parquet_file
from service_framework.deploy_env_config import deploy_env
from service_framework.env_var import EnvVar
from statedb.config.common import (
    ConnectorID,
    IdentityID,
    ProjectID,
    SourceTableName,
    TableIDv2,
    TableLink,
    TableName,
)
from statedb.db.auth import AuthPrincipal
from statedb.db.services import ConnectorService, StateServices, TableService
from statedb.rest.connector_authorization import (
    CONNECTOR_DISCOVERY_ACTIONS,
    CONNECTOR_DOWNLOAD_ACTIONS,
)
from statedb.utils.errors import (
    ConnectorNameConflictError,
    ConnectorSourceTableNotFoundError,
    ConnectorValidationError,
    connector_validation_diagnostic_value,
)
from temporalio_commons.auto_rpc_exception import RpcError
from temporalio_commons.worker_registry import RunSingleActivityWorkflow

log = logging.getLogger(__name__)

S3_VALIDATE_TIMEOUT = 30  # Seconds

# Local files use multipart transport but remain bounded as one completed
# connector object. Keep the prediction-download staging bound separate.
MAX_UPLOADED_FILE_SIZE = 100 * 1024**3  # 100 GiB
MAX_STAGED_DOWNLOAD_SIZE = 1 * 10**9  # 1 GB

_CONNECTOR_ACTIVITY_PUBLIC_ERROR_TYPES = frozenset({
    'ConnectorValidationError',
    # Custom SQL query rejected by the warehouse (bad syntax / unknown
    # function / missing table). A ConnectorValidationError subclass, but this
    # allowlist matches on the Temporal-serialized class *name*, so the
    # subclass must be listed explicitly to keep its warehouse-reason message.
    'CustomSqlQueryError',
    'IdentityResolutionUnavailableError',
    'IdentityValidationError',
    'MultipleIdentitiesValidateTargetError',
    'NoIdentityValidatesTargetError',
})
_CONNECTOR_ACTIVITY_EXACT_PUBLIC_ERROR_TYPES: Dict[object, Dict[str, type]] = {
    create_project_connector_activity: {
        ConnectorNameConflictError.__name__: ConnectorNameConflictError,
    },
}
_CONNECTOR_ACTIVITY_FAILURE_MESSAGE = (
    'Connector operation is temporarily unavailable. Please retry.')

# Interactive connector reads run in a Temporal workflow with this execution
# deadline. Operators can raise it for slow providers or large sources.
_INTERACTIVE_CONNECTOR_READ_TIMEOUT_ENV = EnvVar(
    name='INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS',
    type=float,
    description='Timeout in seconds for connector source-column and '
    'sample-row workflows.',
    default_value=120.0,
)
_INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS = (
    _INTERACTIVE_CONNECTOR_READ_TIMEOUT_ENV.parse_value())
_CONNECTOR_RESULT_DELIVERY_GRACE_SECONDS = 5.0


def _connector_read_timeout_message(timeout: float) -> str:
    budget = f'{timeout:g}'
    return (
        f'Connector source read exceeded the configured {budget}-second '
        'budget. Set INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS to a larger '
        'value for this source, and keep any fronting proxy read timeout '
        'greater than the configured connector budget.')


def _is_connector_read_workflow_timeout(error: BaseException) -> bool:
    return (isinstance(error, WorkflowFailureError)
            and isinstance(error.cause, TemporalTimeoutError)
            and error.cause.type == TimeoutType.START_TO_CLOSE)


def _connector_activity_workflow_id_prefix(activity_fn) -> str:
    return f'connector-operation:{activity_fn.__name__}:'


class ConnectorApiDriver:
    def __init__(
        self,
        data_backend_ctx: DataBackendContext,
        supported_connector_types: List[ConnectorType],
        allow_dev_file_paths: bool = False,  # enabled on dev
        data_scale_limits: Optional[DataScaleLimits] = None,
        state_services: Optional[StateServices] = None,
    ):
        self._data_backend_ctx = data_backend_ctx
        self._supported_connector_types = supported_connector_types
        self._allow_dev_file_paths = allow_dev_file_paths
        self._data_scale_limits = data_scale_limits
        self._state_services = state_services

    def _configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services

    def _require_state_services(self) -> StateServices:
        if self._state_services is None:
            raise RuntimeError("ConnectorApiDriver requires StateServices")
        return self._state_services

    @property
    def _connector_service(self) -> ConnectorService:
        return self._require_state_services().connectors

    @property
    def _table_service(self) -> TableService:
        return self._require_state_services().tables

    @property
    def _uploaded_file_service(self):
        return self._require_state_services().uploaded_files

    async def _connector_reference_for_project(
        self,
        connector_id: ConnectorID,
        project_id: ProjectID,
        principal: AuthPrincipal,
        actions=CONNECTOR_DISCOVERY_ACTIONS,
    ) -> ConnectorExecutionRef:
        return await self._connector_service.execution_ref_for_project(
            principal, project_id, connector_id, actions)

    async def _validated_source_coordinate(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
        source_table_name: str,
    ) -> tuple[str, api.ConnectorResponse | None]:
        """Authorize the connector and enforce its coordinate mode.

        This runs before scheduling provider work. Rooted connectors and
        project uploads accept connector-relative names; rootless object-store
        connectors require a canonical full provider URI.
        """
        connector = await self._connector_service.source_table_connector_for_project(
            project_id, connector_id, principal)
        config = None if connector is None else connector.config
        if (isinstance(config, api.S3ConnectorResourceConfig)
                and isinstance(config.target, api.S3AnyURIConnectorTarget)):
            return normalize_s3_source_uri(source_table_name), connector
        if isinstance(config, api.S3ConnectorResourceConfig):
            return normalize_s3_source_relative_path(
                source_table_name), connector
        file_configs = (api.GCSConnectorResourceConfig,
                        api.ADLSConnectorResourceConfig)
        if isinstance(config, file_configs):
            source = object_store_source_for_connector(config,
                                                       source_table_name)
            coordinate = (source.path if isinstance(
                source, (source_api.GCSRelativeSource,
                         source_api.ADLSRelativeSource)) else source.uri)
            return coordinate, connector
        if isinstance(config, api.DatabricksConnectorResourceConfig):
            try:
                schema_name, table_name = normalize_table_coordinate(
                    source_table_name,
                    catalog=config.catalog,
                    schema=config.schema_name,
                )
            except ValueError as error:
                raise ConnectorError(str(error)) from None
            # Persist a dot-qualified provider coordinate. This keeps the raw
            # source name unambiguous when it legally contains Kumo's
            # ``__kumo__`` output-coordinate separator.
            return f'{schema_name}.{table_name}', connector
        validate_connector_relative_source_name(source_table_name)
        return source_table_name, connector

    async def create_project_connector(
        self,
        project_id: ProjectID,
        args: api.CreateConnectorArgs,
        principal: AuthPrincipal,
    ) -> api.ConnectorResponse:
        authorizations = await self._connector_service.authorize_create(
            project_id, args, principal)
        response, _ = await self._execute_connector_activity(
            create_project_connector_activity,
            [project_id, args, authorizations],
        )
        return response

    async def list_project_connectors(
        self,
        project_id: ProjectID,
        principal: AuthPrincipal,
    ) -> list[api.ConnectorResponse]:
        return await self._connector_service.list_for_project(
            project_id, principal)

    async def get_project_connector(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
    ) -> api.ConnectorResponse:
        return await self._connector_service.get_for_project(
            project_id, connector_id, principal)

    async def update_project_connector_metadata(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        args: api.UpdateConnectorMetadataArgs,
        principal: AuthPrincipal,
    ) -> api.ConnectorResponse:
        return await self._connector_service.update_metadata(
            project_id, connector_id, args, principal)

    async def delete_project_connector(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
    ) -> api.ConnectorResponse:
        return await self._connector_service.delete(project_id, connector_id,
                                                    principal)

    async def get_or_create_project_upload_connector(
        self,
        project_id: ProjectID,
        principal: AuthPrincipal,
    ) -> ConnectorID:
        await self._connector_service.require_project_access(
            principal, project_id)
        upload_root = osp.join(
            deploy_env().data_plane_storage_root,
            'uploaded-datasets',
            str(project_id),
        )
        await asyncio.to_thread(fsutil.makedirs, upload_root, exist_ok=True)
        return await self._connector_service.register_project_upload_connector(
            project_id, upload_root)

    async def test_identity_credentials(
        self,
        identity_id: IdentityID,
        principal: AuthPrincipal,
    ) -> api.IdentityResponse:
        authorization = await self._require_state_services(
        ).identities.authorize_test_credentials(identity_id, principal)
        response, _ = await self._execute_connector_activity(
            validate_identity_credentials_activity,
            [identity_id, authorization],
        )
        return response

    async def _execute_connector_activity(
        self,
        activity_fn,
        activity_inputs: list,
        *,
        response_token: str = '',
        asynchronous: bool = False,
        timeout: float | None = None,
    ):
        exact_public_error_types: Dict[str, type] = {}
        waiting_for_result = False
        try:
            client = await self._data_backend_ctx.temporal_client().get()
            if response_token:
                handle = client.get_workflow_handle(response_token)
                description = await handle.describe()
                if description.status == WorkflowExecutionStatus.RUNNING:
                    return None, response_token
                if response_token.startswith(
                        _connector_activity_workflow_id_prefix(activity_fn)):
                    exact_public_error_types = (
                        _CONNECTOR_ACTIVITY_EXACT_PUBLIC_ERROR_TYPES.get(
                            activity_fn, {}))
                waiting_for_result = True
                result = handle.result()
                if timeout is not None:
                    result = asyncio.wait_for(
                        result,
                        timeout=(timeout +
                                 _CONNECTOR_RESULT_DELIVERY_GRACE_SECONDS),
                    )
                return await result, ''

            workflow_id = (_connector_activity_workflow_id_prefix(activity_fn) +
                           uuid.uuid4().hex)
            kwargs = dict(
                args=[activity_fn.__name__, activity_inputs],
                id=workflow_id,
                task_queue=DATA_BACKEND_TASK_QUEUE,
            )
            if timeout is not None:
                kwargs['execution_timeout'] = timedelta(seconds=timeout)
            if asynchronous:
                await client.start_workflow(RunSingleActivityWorkflow.run,
                                            **kwargs)
                return None, workflow_id
            execution = client.execute_workflow(RunSingleActivityWorkflow.run,
                                                **kwargs)
            exact_public_error_types = (
                _CONNECTOR_ACTIVITY_EXACT_PUBLIC_ERROR_TYPES.get(
                    activity_fn, {}))
            waiting_for_result = True
            if timeout is not None:
                execution = asyncio.wait_for(
                    execution,
                    timeout=(timeout +
                             _CONNECTOR_RESULT_DELIVERY_GRACE_SECONDS),
                )
            return await execution, ''
        except asyncio.CancelledError:
            raise
        except Exception as error:
            read_timed_out = (timeout is not None and waiting_for_result
                              and _is_connector_read_workflow_timeout(error))
            allowed_error_types = (_CONNECTOR_ACTIVITY_PUBLIC_ERROR_TYPES
                                   | frozenset(exact_public_error_types))
            public_error = (data_backend_module.
                            _extract_allowlisted_temporal_public_exception(
                                error,
                                allowed_error_types=allowed_error_types,
                                allow_timeout=False,
                                exact_error_types=exact_public_error_types,
                            ))
            if public_error is not None:
                failure = public_error
            else:
                failure = None
        # Translate outside the active exception context. Unknown Temporal,
        # worker, activity-timeout, and transport failures remain sanitized
        # 5xx errors. Exact allowlisted public envelopes retain their 4xx/503,
        # and the configured workflow execution deadline maps to 422 below.
        if failure is not None:
            raise failure from None
        if read_timed_out:
            assert timeout is not None
            raise ConnectorError(
                _connector_read_timeout_message(timeout),
                non_retryable=True,
            ) from None
        raise RpcError(
            'data_backend',
            activity_fn.__name__,
            'InternalError',
            _CONNECTOR_ACTIVITY_FAILURE_MESSAGE,
            '',
            False,
        ) from None

    async def get_project_source_tables(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
        *,
        response_token: str = '',
        schema_name: str = '',
        sync: bool = False,
    ) -> SourceTablesResponse:
        connector = await self._connector_service.source_table_connector_for_project(
            project_id, connector_id, principal)
        if (connector is not None and isinstance(
                connector.config,
            (api.S3ConnectorResourceConfig, api.GCSConnectorResourceConfig,
             api.ADLSConnectorResourceConfig)) and isinstance(
                 connector.config.target,
                 (api.S3AnyURIConnectorTarget, api.GCSAnyURIConnectorTarget,
                  api.ADLSAnyURIConnectorTarget))):
            raise ConnectorError(
                'Any-URI connectors require an exact full table URI and do '
                'not support listing')
        reference = await self._connector_reference_for_project(
            connector_id, project_id, principal)
        existing = await self._table_service.list_tables(
            project_id, connector_id=connector_id)
        response, pending_token = await self._execute_connector_activity(
            list_connector_source_tables,
            [reference, existing, schema_name or None],
            response_token=response_token,
            asynchronous=not sync,
        )
        return (response if response is not None else SourceTablesResponse(
            response_token=pending_token))

    async def get_project_source_table_names(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
    ) -> list[str]:
        response = await self.get_project_source_tables(
            project_id,
            connector_id,
            principal,
            sync=True,
        )
        return [table.name for table in response.source_tables]

    async def get_project_source_table_file_type(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
        source_table_name: str,
    ) -> tuple[str | None, str | None]:
        """Resolve file type in an activity without request-side credentials."""
        source_table_name, _ = await self._validated_source_coordinate(
            project_id, connector_id, principal, source_table_name)
        reference = await self._connector_reference_for_project(
            connector_id, project_id, principal)
        response, _ = await self._execute_connector_activity(
            get_connector_source_table_file_type,
            [reference, source_table_name],
        )
        return response

    async def get_project_source_download_urls(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
        source_table_name: str,
    ) -> list[str]:
        source_table_name, _ = await self._validated_source_coordinate(
            project_id, connector_id, principal, source_table_name)
        reference = await self._connector_reference_for_project(
            connector_id,
            project_id,
            principal,
            CONNECTOR_DOWNLOAD_ACTIONS,
        )
        staging_path, _ = await self._execute_connector_activity(
            stage_connector_source_download,
            [reference, source_table_name,
             uuid.uuid4().hex],
        )
        if not isinstance(staging_path, str):
            raise ConnectorValidationError('Connector source staging failed')
        return await presign_parquet_table_data.coroutine(staging_path)

    async def get_project_source_table_config(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
        source_table_name: str,
    ) -> source_api.SourceTableConfigResponse:
        """Resolve one exact source through the authorization boundary.

        External credentials remain activity-only: request handling receives
        credential-free connector metadata, while object-store file type is
        resolved by ``get_project_source_table_file_type``. Warehouse sources
        are checked for exact readability without using cached metadata;
        missing or provider-ambiguous sources raise ``ValueError`` as invalid
        table input, while unrelated connector failures propagate.
        """
        source_table_name, connector = await self._validated_source_coordinate(
            project_id, connector_id, principal, source_table_name)
        config = None if connector is None else connector.config

        if isinstance(config, api.SnowflakeConnectorResourceConfig):
            await self._require_project_warehouse_source_table(
                project_id, connector_id, principal, source_table_name)
            source_table = source_api.SnowflakeSourceTable(
                snowflake_connector_id=str(connector_id),
                database=config.database,
                schema_name=config.schema_name,
                table=source_table_name,
            )
        elif isinstance(config, api.DatabricksConnectorResourceConfig):
            await self._require_project_warehouse_source_table(
                project_id, connector_id, principal, source_table_name)
            source_table = source_api.DatabricksSourceTable(
                databricks_connector_id=str(connector_id),
                table=source_table_name,
            )
        elif isinstance(config, api.BigQueryConnectorResourceConfig):
            await self._require_project_warehouse_source_table(
                project_id, connector_id, principal, source_table_name)
            source_table = source_api.BigQuerySourceTable(
                bigquery_connector_id=str(connector_id),
                table_name=source_table_name,
                project_id=config.project_id,
                dataset_id=config.dataset_id,
            )
        else:
            file_type = await self._get_project_source_table_file_type(
                project_id,
                connector_id,
                principal,
                source_table_name,
                config,
            )
            common = {
                'connector_id': str(connector_id),
                'file_type': file_type,
            }
            if config is None:
                root = deploy_env().data_plane_storage_root
                if fsutil.is_gcs(root):
                    source_table = source_api.GCSSourceTable(
                        **common,
                        source=source_api.GCSRelativeSource(
                            path=source_table_name),
                    )
                elif fsutil.is_adls(root):
                    source_table = source_api.ADLSSourceTable(
                        **common,
                        source=source_api.ADLSRelativeSource(
                            path=source_table_name),
                    )
                else:
                    source_table = source_api.S3SourceTable(
                        **common,
                        source=source_api.S3RelativeSource(
                            path=source_table_name),
                    )
            elif isinstance(config, api.S3ConnectorResourceConfig):
                if isinstance(config.target, api.S3RootedConnectorTarget):
                    source = source_api.S3RelativeSource(path=source_table_name)
                elif isinstance(config.target, api.S3AnyURIConnectorTarget):
                    source = source_api.S3URISource(uri=source_table_name)
                else:
                    raise ValueError('Unsupported S3 connector target')
                source_table = source_api.S3SourceTable(**common, source=source)
            elif isinstance(config, api.GCSConnectorResourceConfig):
                source = object_store_source_for_connector(
                    config, source_table_name)
                source_table = source_api.GCSSourceTable(
                    **common, source=source)
            elif isinstance(config, api.ADLSConnectorResourceConfig):
                source = object_store_source_for_connector(
                    config, source_table_name)
                source_table = source_api.ADLSSourceTable(
                    **common, source=source)
            else:
                raise ValueError(
                    'Connector does not expose file-backed source tables')

        return source_api.SourceTableConfigResponse(source_table=source_table)

    async def _require_project_warehouse_source_table(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
        source_table_name: str,
    ) -> None:
        """Reauthorize and validate one exact warehouse table without cache.

        Missing and provider-ambiguous sources fail as invalid table input;
        unrelated connector failures propagate.
        """
        reference = await self._connector_reference_for_project(
            connector_id, project_id, principal)
        exists, _ = await self._execute_connector_activity(
            validate_connector_source_table_exists,
            [reference, source_table_name],
        )
        if not exists:
            raise ValueError(f'Invalid table {source_table_name}')

    async def _get_project_source_table_file_type(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
        source_table_name: str,
        config: api.ConnectorResourceConfig | None,
    ) -> source_api.FileType:
        if config is None:
            uploaded = await self._uploaded_file_service.get_completed(
                project_id, connector_id, source_table_name)
            if uploaded is None:
                raise KeyError(f'Unable to find table {source_table_name}')
            return source_api.FileType(uploaded.file_type.value.upper())

        if not isinstance(config, (
                api.S3ConnectorResourceConfig,
                api.GCSConnectorResourceConfig,
                api.ADLSConnectorResourceConfig,
        )):
            raise ValueError(
                'Connector does not expose file-backed source tables')
        file_type_value, error = await self.get_project_source_table_file_type(
            project_id,
            connector_id,
            principal,
            source_table_name,
        )
        if error == 'not_found':
            raise KeyError(f'Unable to find table {source_table_name}')
        if error is not None or file_type_value is None:
            raise ValueError(
                'Connector does not expose file-backed source tables')
        return source_api.FileType(file_type_value)

    async def get_project_source_columns(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
        source_table_name: str,
        *,
        skip_cache: bool = True,
        timeout: float | None = _INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS,
        query: str | None = None,
    ) -> SourceColumnsResource:
        """Return columns for a physical source or Custom SQL result.

        Missing and provider-ambiguous physical sources retain the connector
        discovery 404 contract across the activity boundary.
        """
        # A Custom SQL ``query`` discovers columns from its described result
        # schema (no data scan), so it has no source coordinate to validate.
        if query is None:
            source_table_name, _ = await self._validated_source_coordinate(
                project_id, connector_id, principal, source_table_name)
        reference = await self._connector_reference_for_project(
            connector_id, project_id, principal)
        # Append ``query`` only in query mode so the physical path retains its
        # prior activity arity during rolling worker deployments.
        activity_inputs = [reference, source_table_name, skip_cache]
        if query is not None:
            activity_inputs.append(query)
        response, _ = await self._execute_connector_activity(
            get_connector_source_columns,
            activity_inputs,
            timeout=timeout,
        )
        return response

    async def get_project_sample_rows(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
        source_table_name: str,
        *,
        num_rows: int = DEFAULT_NUM_SAMPLE_ROWS,
        serialize_to_json: bool = False,
        timeout: float | None = _INTERACTIVE_CONNECTOR_READ_TIMEOUT_SECONDS,
        skip_cache: bool = True,
        response_token: str = '',
        is_async: bool = False,
        require_stats: bool = True,
        query: str | None = None,
    ) -> SampleRowsResponse:
        """Return sample rows for a physical source or Custom SQL result.

        Missing and provider-ambiguous physical sources retain the connector
        discovery 404 contract across the activity boundary.
        """
        # A Custom SQL ``query`` returns a bounded synchronous preview and has
        # no source coordinate or asynchronous response-token flow.
        if query is None:
            source_table_name, _ = await self._validated_source_coordinate(
                project_id, connector_id, principal, source_table_name)
        reference = await self._connector_reference_for_project(
            connector_id, project_id, principal)
        # Append ``query`` only in query mode so the physical path retains its
        # prior activity arity during rolling worker deployments.
        activity_inputs = [
            reference,
            source_table_name,
            num_rows,
            serialize_to_json,
            skip_cache,
            require_stats,
        ]
        if query is not None:
            activity_inputs.append(query)
        response, pending_token = await self._execute_connector_activity(
            get_connector_sample_rows,
            activity_inputs,
            response_token=response_token if query is None else '',
            asynchronous=is_async and query is None,
            timeout=timeout,
        )
        return (response if response is not None else SampleRowsResponse(
            response_token=pending_token))

    async def is_project_source_table_view(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
        source_table_name: str,
    ) -> bool:
        reference = await self._connector_reference_for_project(
            connector_id, project_id, principal)
        result, _ = await self._execute_connector_activity(
            is_connector_source_table_view,
            [reference, source_table_name],
        )
        return result

    async def validate_project_connector(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        principal: AuthPrincipal,
    ) -> None:
        reference = await self._connector_reference_for_project(
            connector_id, project_id, principal)
        await self._execute_connector_activity(
            validate_connector_reference,
            [reference],
        )

    @blocking_method(fut_dedupe_option=FutureDedupeOption.DISABLED)
    def _upload_file_part(
        self,
        project_id: ProjectID,
        temp_upload_path: str,
        upload_id: str,
        part_number: int,
        body: bytes,
    ) -> str:
        root_dir = deploy_env().data_plane_storage_root
        if not fsutil.is_local(root_dir):
            raise ValueError(
                FILE_UPLOAD_NOT_SUPPORTED_MESSAGE.format(root_dir=root_dir))
        validate_local_upload_path(temp_upload_path, root_dir)
        rel = osp.relpath(osp.abspath(temp_upload_path), osp.abspath(root_dir))
        project_prefix = osp.join('temporary-uploaded-datasets',
                                  str(project_id)) + osp.sep
        if not rel.startswith(project_prefix):
            raise ValueError(
                "Upload path must be under the active project's temporary "
                "upload directory")
        part_path = local_upload_part_path(temp_upload_path, upload_id,
                                           part_number)
        fsutil.makedirs(osp.dirname(part_path), exist_ok=True)
        fsutil.write_binary(part_path, body)
        return hashlib.md5(body, usedforsecurity=False).hexdigest()

    async def upload_file_part(
        self,
        project_id: ProjectID,
        principal: AuthPrincipal,
        temp_upload_path: str,
        upload_id: str,
        part_number: int,
        body: bytes,
    ) -> str:
        await self._connector_service.require_project_access(
            principal, project_id)
        return await self._upload_file_part.coroutine(
            project_id,
            temp_upload_path,
            upload_id,
            part_number,
            body,
        )

    async def start_file_upload(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        request: api.StartFileUploadRequest,
    ) -> api.StartFileUploadResponse:
        root_dir = deploy_env().data_plane_storage_root
        if not supports_backend_file_upload(root_dir):
            raise ValueError(
                FILE_UPLOAD_NOT_SUPPORTED_MESSAGE.format(root_dir=root_dir))
        connector = await self._connector_service.get_internal(
            project_id, connector_id)
        if not isinstance(connector.config, FileConnectorResourceConfig):
            raise ValueError('Project upload connector is invalid')
        source_table_name = SourceTableName(request.source_table_name)

        # Upload the file to a temporary directory in data-plane-bucket:
        suffix = self._upload_suffix(request.file_type)
        upload_path = osp.join(
            root_dir,
            "temporary-uploaded-datasets",
            str(project_id),
            uuid.uuid4().hex,
            str(source_table_name) + suffix,
        )
        helper = fsutil.get_presigned_helper(upload_path)
        upload_id, presigned_urls = (await
                                     helper.create_multipart_upload.coroutine(
                                         request.num_parts))

        return api.StartFileUploadResponse(
            temp_upload_path=upload_path,
            upload_id=upload_id,
            presigned_part_urls=presigned_urls,
        )

    @blocking_method()
    def _validate_uploaded_file(self, file_path: str, *,
                                validate_data: bool = True):
        log.debug(f"Validating uploaded file at {file_path}")
        # Enforce the upload contract even when a caller skips schema/data
        # validation.
        file_size = fsutil.filesize(file_path) or 0
        if file_size > MAX_UPLOADED_FILE_SIZE:
            max_size = humanize.naturalsize(MAX_UPLOADED_FILE_SIZE, binary=True)
            raise ValueError(FILE_SIZE_EXCEEDED_MESSAGE.format(max_size))

        if not validate_data:
            return

        suffix = osp.splitext(file_path)[1]
        if suffix == ".csv":
            validate_csv_file(file_path)
        else:
            validate_parquet_file(file_path)

    async def complete_file_upload(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        request: api.CompleteFileUploadRequest,
    ) -> None:
        root_dir = deploy_env().data_plane_storage_root
        if not supports_backend_file_upload(root_dir):
            raise ValueError(
                FILE_UPLOAD_NOT_SUPPORTED_MESSAGE.format(root_dir=root_dir))
        connector = await self._connector_service.get_internal(
            project_id, connector_id)
        if not isinstance(connector.config, FileConnectorResourceConfig):
            raise ValueError('Project upload connector is invalid')
        source_table_name = SourceTableName(request.source_table_name)
        suffix = self._upload_suffix(request.file_type)
        temporary_root = osp.join(root_dir, 'temporary-uploaded-datasets',
                                  str(project_id))
        self._validate_scoped_upload_path(request.temp_upload_path,
                                          temporary_root)
        if osp.basename(
                request.temp_upload_path) != str(source_table_name) + suffix:
            raise ValueError('Temporary upload path does not match request')

        @blocking_func()
        def remove_file(path: str) -> None:
            fsutil.rm(path, ignore_errors=True)

        helper = fsutil.get_presigned_helper(request.temp_upload_path)
        await helper.complete_multipart_upload.coroutine(
            request.upload_id,
            request.parts_metadata,
        )

        try:
            await self._validate_uploaded_file.coroutine(
                request.temp_upload_path,
                validate_data=request.validate_data,
            )
        except Exception as e:
            # Delete the file stored at the temporary path:
            log.info("Validation failed for the uploaded file: "
                     f"{request.temp_upload_path}. Deleting the file.")
            log.exception(e)
            await remove_file.coroutine(request.temp_upload_path)
            raise e

        @blocking_func()
        def post_process_file(temp_path: str, dest_path: str):
            log.info(
                f"Copying validated upload file from {temp_path} to {dest_path}"
            )
            fsutil.copy(temp_path, dest_path, overwrite=True,
                        raise_on_exists=False)

        dest_path = osp.join(
            connector.config.root_dir,
            str(source_table_name) + suffix,
        )
        await post_process_file.coroutine(request.temp_upload_path, dest_path)
        await self._uploaded_file_service.record_completed(
            project_id,
            connector_id,
            source_table_name,
            request.file_type,
            dest_path,
        )
        await remove_file.coroutine(request.temp_upload_path)

    async def delete_uploaded_file(
        self,
        project_id: ProjectID,
        connector_id: ConnectorID,
        request: api.DeleteUploadedFileRequest,
    ) -> bool:
        r"""
        Returns True if the file was deleted, False if the file did not exist.
        """
        connector_entry = await self._connector_service.get_internal(
            project_id, connector_id)
        config = connector_entry.config
        if not isinstance(config, FileConnectorResourceConfig):
            raise ValueError('Project upload connector is invalid')
        source_table_name = SourceTableName(request.source_table_name)
        uploaded_file = await self._uploaded_file_service.get(
            project_id,
            connector_id,
            source_table_name,
            request.file_type,
        )
        if uploaded_file is None:
            return False
        file_path = uploaded_file.path
        log.debug(f"Deleting file at {file_path}")
        if fsutil.exists(file_path):
            fsutil.rm(file_path)
        await self._uploaded_file_service.delete(
            project_id,
            connector_id,
            source_table_name,
            request.file_type,
        )
        return True

    @staticmethod
    def _upload_suffix(file_type: api.UploadFileType) -> str:
        return ('.parquet'
                if file_type == api.UploadFileType.PARQUET else '.csv')

    @staticmethod
    def _validate_scoped_upload_path(path: str, root_dir: str) -> None:
        if fsutil.is_local(root_dir):
            validate_local_upload_path(path, root_dir)
            return
        root_prefix = root_dir.rstrip('/') + '/'
        if not path.startswith(root_prefix):
            raise ValueError('Upload path is outside the active project')
        relative_path = path[len(root_prefix):]
        if '..' in relative_path.split('/'):
            raise ValueError('Upload path is outside the active project')

    @staticmethod
    def _connector_config_for_sdk(
        connector: Connector, ) -> Optional[Dict[str, str]]:
        if isinstance(connector, SnowflakeConnector):
            return {
                "name": connector.name,
                "account": connector.account,
                "warehouse": connector.warehouse,
                "database": connector.database,
                "schema_name": connector.schema_name,
                "data_source_type": DataSourceType.SNOWFLAKE.value,
            }
        if isinstance(connector, DatabricksConnector):
            return {
                "name": connector.name,
                "host": connector.host,
                "cluster_id": connector.cluster_id,
                "warehouse_id": connector.warehouse_id,
                "catalog": connector.catalog,
                "data_source_type": DataSourceType.DATABRICKS.value,
            }
        if isinstance(connector, BigQueryConnector):
            return {
                "name": connector.name,
                "project_id": connector.project_id,
                "dataset_id": connector.dataset_id,
                "data_source_type": DataSourceType.BIGQUERY.value,
            }
        raise ValueError("Unsupported connector type "
                         f"{connector._get_connector_type()} to create Ui to "
                         "SDK script.")

    async def get_bp_connector_metadata_for_sdk(
        self,
        connector_metadata: Dict[str, str],
        project_id: ProjectID,
        principal: AuthPrincipal,
    ) -> Optional[Dict[str, str]]:
        """
        Used to generate an SDK-compatible connector config for a
        batch prediction export connector.
        This method is part of the UI to SDK feature.
        Args:
            connector_metadata (Dict[str, str])
        Response:
            Dict[str, str]: SDK-compatible connector config.
        """
        if not connector_metadata:
            return None
        connector_id = connector_metadata.get("connector_id")
        if connector_id is None:
            raise ValueError(
                "Connector ID is required in connector_metadata to fetch metadata."
            )
        connector = await self._connector_service.get_for_project(
            project_id,
            ConnectorID(connector_id),
            principal,
        )
        expected_type = connector_metadata.get('data_source_type')
        actual_type = connector.config.type.value.upper()
        if expected_type != actual_type:
            raise ValueError(f"Output connector '{connector_id}' has type "
                             f"'{actual_type}', not '{expected_type}'.")
        return {
            'connector_id': str(connector.id),
            'lookup_name': connector.name,
            'data_source_type': actual_type,
        }


def _source_table_from_config_and_existing_tables(
    table_name: str,
    existing_v2_tables: List[TableDBEntryV2],
    connector: Connector,
) -> SourceTable:
    downstream_table_ids: List[TableLink] = []

    # V2 tables created through /v1/tables/ live in a separate store and expose
    # their source through TableDefinition.get_source_table_name(); link them.
    for entry in existing_v2_tables:
        if str(entry.config.get_source_table_name()) == table_name:
            alias = entry.alias or str(entry.config.get_source_table_name())
            downstream_table_ids.append(
                TableLink(
                    name=alias,
                    id=TableIDv2.public(TableName(alias), entry.id, None),
                ))

    path = None
    if isinstance(connector, CompositeFileConnector):
        try:
            path = connector._get_path(table_name)
        except FileNotFoundError:
            # TODO(manan): this should never happen in production
            pass
    return SourceTable(
        name=table_name,
        num_cols=0,
        downstream_table_ids=downstream_table_ids,
        path=path,
    )


@blocking_func(DATA_BACKEND_THREAD_POOL)
def _get_source_tables_raw(
    id: ConnectorID,
    connector: Connector,
    existing_v2_tables: List[TableDBEntryV2],
    resolved_spec: ConnectorOpenSpec,
    schema_name: Optional[str] = None,
) -> SourceTablesResponse:
    r"""Get a list of source tables from the connector.

    Args:
        id (ConnectorID): The connector ID.
        connector (Connector): Connector where the table is stored.
        existing_v2_tables (List[TableDBEntryV2]): Existing v2 tables in the
            database, used to populate downstream table links.
        schema_name (Optional[str]): The schema name to get source tables
            for.

    Returns:
        SourceTablesResponse: contains either a list of source tables or
            a response token that used to retrieve source tables.
    """
    ret = SourceTablesResponse()
    try:
        if isinstance(connector, AnyURIS3Connector):
            raise ConnectorError(
                'Any-URI S3 connectors require an exact s3:// table URI and '
                'do not support listing')
        empty_connector = False
        is_project_upload = (isinstance(resolved_spec, ConnectorSpec)
                             and resolved_spec.is_project_upload)
        if is_project_upload:
            if not isinstance(connector, FileConnector):
                raise ConnectorValidationError(
                    'Project upload connector has an invalid configuration')
            # A newly provisioned project upload connector has an empty root.
            # This is the only connector allowed to inspect its root outside
            # an external identity credential scope. The attestation is
            # derived from canonical active StateDB metadata in this activity.
            files_list = fsutil.listdir(connector.root_dir)
            log.debug(f"Files list for connector {id}: {files_list}")
            if len(files_list) == 0:
                empty_connector = True

        if not empty_connector:
            table_names = connector.list_tables(schema_name)
            tables = [
                _source_table_from_config_and_existing_tables(
                    t, existing_v2_tables, connector) for t in table_names
            ]
            ret = SourceTablesResponse(
                source_tables=sorted(tables, key=lambda table: table.name),
                response_token="",
            )

    except ConnectorValidationError:
        raise
    except Exception as e:
        # Provider exceptions can embed signed assertions, bearer tokens, or
        # request bodies. Keep diagnostics useful without copying exception
        # text into application logs.
        log.warning(
            "Connector source discovery failed (error_type=%s)",
            type(e).__name__,
        )
        raise ConnectorValidationError(
            'Connector source discovery failed') from None

    return ret


@connector_activity_registry.connector_authorized_activity
def list_connector_source_tables(
    reference: ConnectorExecutionRef,
    existing_v2_tables: List[TableDBEntryV2],
    schema_name: Optional[str] = None,
) -> SourceTablesResponse:
    connector = None
    try:
        try:
            from rest.backends.data_backend import connector_broker

            # This non-Temporal spec is composed from the active StateDB row
            # inside the activity. No caller-provided flag can grant internal
            # upload behavior to an external or stale connector reference.
            resolved_spec = (
                connector_broker.connector_credential_broker.resolve(reference))
            connector = get_connector_from_spec(resolved_spec)
            return _get_source_tables_raw(
                reference.connector_id,
                connector,
                existing_v2_tables,
                resolved_spec,
                schema_name,
            )
        finally:
            if connector is not None:
                connector.close()
    except ConnectorValidationError:
        raise
    except Exception as error:
        log.warning('Connector source discovery failed (error_type=%s)',
                    type(error).__name__)
        raise ConnectorValidationError(
            'Connector source discovery failed') from None


@connector_activity_registry.connector_authorized_activity
def get_connector_source_table_file_type(
    reference: ConnectorExecutionRef,
    source_table_name: str,
) -> tuple[str | None, str | None]:
    connector = None
    try:
        try:
            connector = get_connector_from_spec(reference)
            if not isinstance(connector, CompositeFileConnector):
                return None, 'unsupported'
            try:
                _, file_type = connector.get_path_and_type(source_table_name)
            except ValidationException:
                raise
            except FileNotFoundError:
                return None, 'not_found'
            except Exception as error:
                log.warning(
                    'Object-store source type resolution failed '
                    '(error_type=%s)',
                    type(error).__name__)
                return None, 'access_failed'
            return str(file_type).upper(), None
        finally:
            if connector is not None:
                connector.close()
    except (ConnectorValidationError, ValidationException):
        raise
    except Exception as error:
        log.warning('Connector source type resolution failed (error_type=%s)',
                    type(error).__name__)
        raise ConnectorValidationError(
            'Connector source type resolution failed') from None


@connector_activity_registry.connector_authorized_activity
def stage_connector_source_download(
    reference: ConnectorExecutionRef,
    source_table_name: str,
    staging_id: str,
) -> str:
    connector = None
    try:
        connector = get_connector_from_spec(reference)
        if not isinstance(connector, CompositeFileConnector):
            raise ConnectorValidationError(
                'Download references require an object-store connector')
        if not staging_id or not staging_id.isalnum():
            raise ConnectorValidationError('Invalid download staging ID')
        if isinstance(connector, AnyURIFileConnector):
            source_table_name = connector._normalizer(source_table_name)
        else:
            validate_connector_relative_source_name(source_table_name)
        with connector.filesystem_credential_scope(source_table_name):
            path = connector._get_path(source_table_name)
            files = (_list_dataset_files(path, ('csv',
                                                'parquet'), MAX_SUPPORTED_FILES)
                     if fsutil.isdir(path) else [path])
            if not files or any(
                    connector._supported_file_type(file) != 'parquet'
                    for file in files):
                raise ConnectorValidationError(
                    'Prediction-table readback requires Parquet data')
        with connector._local_source(
                source_table_name, transform=False,
                max_entries=MAX_SUPPORTED_FILES,
                max_bytes=MAX_STAGED_DOWNLOAD_SIZE) as (staged, _):
            staging_path = osp.join(
                deploy_env().data_plane_storage_root,
                'temporary-uploaded-datasets',
                str(reference.project_id),
                'downloads',
                staging_id,
                'table',
            )
            fsutil.rm(staging_path, ignore_errors=True)
            fsutil.makedirs(osp.dirname(staging_path), exist_ok=True)
            fsutil.copy(staged.directory, staging_path, overwrite=True,
                        raise_on_exists=False)
            return staging_path
    except ConnectorValidationError:
        raise
    except Exception as error:
        log.warning('Connector source staging failed (error_type=%s)',
                    type(error).__name__)
        raise ConnectorValidationError(
            'Connector source staging failed') from None
    finally:
        if connector is not None:
            connector.close()


@connector_activity_registry.connector_authorized_activity
def get_connector_source_columns(
    reference: ConnectorExecutionRef,
    source_table_name: str = '',
    skip_cache: bool = True,
    query: str | None = None,
) -> SourceColumnsResource:
    """Return source columns while preserving discovery error contracts.

    Missing and provider-ambiguous physical sources become
    ``ConnectorSourceTableNotFoundError``. Custom SQL and unrelated connector
    failures retain the connector-validation contract.
    """
    # A Custom SQL ``query`` is described for its output columns (no data
    # scan). Only that warehouse path needs connector-dialect resolution for
    # the advisory read-only pre-filter; physical tables keep the cheap path.
    connector = None
    try:
        try:
            if query is None:
                connector = get_connector_from_spec(reference)
                return get_columns_response(
                    connector,
                    source_table_name,
                    skip_cache=skip_cache,
                )
            connector, connector_type = resolve_connector_and_type(reference)
            return get_query_columns_response(connector, query, connector_type)
        finally:
            if connector is not None:
                connector.close()
    except (
            FileNotFoundError,
            ConnectorSourceTableNotFoundError,
            SourceTableNotFoundError,
            SourceTableUnavailableError,
    ):
        if query is not None:
            raise ConnectorValidationError(
                'Connector source access failed') from None
        raise ConnectorSourceTableNotFoundError(
            f'Unable to find table {source_table_name}') from None
    except (ValidationException, ConnectorValidationError) as error:
        if (type(error).__name__ == ConnectorSourceTableNotFoundError.__name__):
            raise ConnectorValidationError(
                'Connector source access failed') from None
        raise
    except Exception as error:
        log.warning('Connector columns read failed (error_type=%s)',
                    type(error).__name__)
        raise ConnectorValidationError(
            'Connector source access failed') from None


@connector_activity_registry.connector_authorized_activity
def validate_connector_source_table_exists(
    reference: ConnectorExecutionRef,
    source_table_name: str,
) -> bool:
    """Validate one table through active activity-only connector credentials.

    Missing and provider-ambiguous source errors return ``False``. Unrelated
    connector or provider failures remain errors instead of being mistaken for
    absence.
    """
    connector = None
    try:
        try:
            connector = get_connector_from_spec(reference)
            return connector.table_exists(source_table_name)
        finally:
            if connector is not None:
                connector.close()
    except (
            FileNotFoundError,
            SourceTableNotFoundError,
            SourceTableUnavailableError,
            ValidationException,
    ):
        return False
    except ConnectorValidationError:
        raise
    except Exception as error:
        log.warning(
            'Connector exact source validation failed (error_type=%s)',
            type(error).__name__,
        )
        raise ConnectorValidationError(
            'Connector source access failed') from None


@connector_activity_registry.connector_authorized_activity
def get_connector_sample_rows(
    reference: ConnectorExecutionRef,
    source_table_name: str,
    num_rows: int,
    serialize_to_json: bool,
    skip_cache: bool,
    require_stats: bool,
    query: str | None = None,
) -> SampleRowsResponse:
    """Return source samples while preserving discovery error contracts.

    Missing and provider-ambiguous physical sources become
    ``ConnectorSourceTableNotFoundError``. Custom SQL and unrelated connector
    failures retain the connector-validation contract.
    """
    # A Custom SQL ``query`` returns a bounded synchronous preview; the
    # physical-table path is unchanged.
    connector = None
    try:
        try:
            if query is None:
                connector = get_connector_from_spec(reference)
                response, _ = get_sample_rows_response(
                    connector,
                    source_table_name,
                    num_rows,
                    serialize_to_json=serialize_to_json,
                    skip_cache=skip_cache,
                    response_token='',
                    is_async=False,
                    require_stats=require_stats,
                )
                return response
            connector, connector_type = resolve_connector_and_type(reference)
            return get_query_sample_rows_response(
                connector, query, connector_type, num_rows,
                serialize_to_json=serialize_to_json)
        finally:
            if connector is not None:
                connector.close()
    except (
            FileNotFoundError,
            ConnectorSourceTableNotFoundError,
            SourceTableNotFoundError,
            SourceTableUnavailableError,
    ):
        if query is not None:
            raise ConnectorValidationError(
                'Connector source access failed') from None
        raise ConnectorSourceTableNotFoundError(
            f'Unable to find table {source_table_name}') from None
    except (ValidationException, ConnectorValidationError) as error:
        if (type(error).__name__ == ConnectorSourceTableNotFoundError.__name__):
            raise ConnectorValidationError(
                'Connector source access failed') from None
        raise
    except Exception as error:
        log.warning('Connector sample read failed (error_type=%s)',
                    type(error).__name__)
        raise ConnectorValidationError(
            'Connector source access failed') from None


_CONNECTOR_ACTIVITY_EXACT_PUBLIC_ERROR_TYPES.update({
    get_connector_source_columns: {
        ConnectorSourceTableNotFoundError.__name__:
        ConnectorSourceTableNotFoundError,
    },
    get_connector_sample_rows: {
        ConnectorSourceTableNotFoundError.__name__:
        ConnectorSourceTableNotFoundError,
    },
})


@connector_activity_registry.connector_authorized_activity
def is_connector_source_table_view(
    reference: ConnectorExecutionRef,
    source_table_name: str,
) -> bool:
    connector = None
    try:
        try:
            connector = get_connector_from_spec(reference)
            return connector.is_table_view(source_table_name)
        finally:
            if connector is not None:
                connector.close()
    except ConnectorValidationError:
        raise
    except Exception as error:
        log.warning('Connector view check failed (error_type=%s)',
                    type(error).__name__)
        raise ConnectorValidationError(
            'Connector source view check failed') from None


@connector_activity_registry.connector_authorized_activity
def validate_connector_reference(reference: ConnectorExecutionRef) -> None:
    from rest.backends.data_backend import connector_broker
    from rest.config.identity import validate_resolved_connector

    try:
        validate_resolved_connector(
            connector_broker.connector_credential_broker.resolve(reference))
    except Exception as error:
        # Warehouse/provider exceptions routinely include assertions, tokens,
        # and request details. Never serialize them through Temporal or expose
        # them in the validation endpoint response; surface only a safe
        # diagnostic category (when the connector classified the failure) plus
        # a correlation id that ties the response to the logged cause. Mirrors
        # the identity-resolver boundary (statedb.db.identities.authorize_create).
        correlation_id = uuid.uuid4().hex
        diagnostic_code = connector_validation_diagnostic_value(error)
        log.warning(
            'Connector target validation failed (correlation_id=%s '
            'error_type=%s diagnostic_code=%s)', correlation_id,
            type(error).__name__, diagnostic_code or 'unclassified')
        category = f' [{diagnostic_code}]' if diagnostic_code else ''
        raise ConnectorValidationError(
            'Connector target validation failed'
            f'{category}; reference: {correlation_id}') from None
