# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import kumoapi.table as table_api
from kumo_core.types import ConnectorType
from kumo_core.types.schema import FPID
from kumoapi.common import ValidationError as ApiValidationError
from temporalio.client import WorkflowExecutionStatus

from kumo.connector.connector import ConnectorError
from kumo.data import get_active_cache
from kumo.data.backend_cache import (
    AsyncDataTaskStatus,
    TableDataTaskResult,
    TableIngestionInfo,
    TableRefreshStage,
)
from kumo.data.custom_sql import (
    advisory_read_only_check,
    validate_connector_sql_scope,
)
from kumo.validation import ValidationException
from kumo.validation.validation import ValidationError, ValidationResponse
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.data_backend import DataBackend
from rest.backends.data_backend.workflows import TableRefreshWorkflow
from rest.config.data_backend import ConnectorExecutionRef, ConnectorSpec
from rest.config.graph_v2 import ListResponse
from rest.config.table import ColStatsResponse
from rest.config.table_v2 import (
    CustomSQLSourceTable,
    TableDBEntryV2,
    TableDefinition,
    TableReferenceMetadata,
    TableResourceResponse,
    derive_table_kind,
)
from rest.deps.context import _current_user_ctx
from rest.driver.error_constants import (
    TABLE_READ_ERROR_MESSAGE,
    TABLE_READ_PROVIDER_ERROR,
)
from rest.driver.table_logic import (
    TableLogic,
    _sql_reference_stype_hints,
    _sql_resolve_request,
    authoritative_sql_connector_type,
    canonicalize_custom_sql_source,
    get_table_col_stats_and_sample_rows_activity,
    infer_table_metadata_activity,
    validate_custom_sql,
    validate_table_definition_activity,
)
from rest.settings import DataScaleLimits
from rest.utils.driver_common import extract_source_table_type
from rest.utils.routers import ValidateNameResponse, ValidationHTTPException
from rest.utils.table import get_full_time_range, get_table_stats
from rest.utils.ui_to_sdk_utils import (
    generate_conn_graph_or_table_sdk_script,
    make_stype_and_dtype_sdk_compat,
)
from statedb.config.common import (
    ID,
    ConnectorID,
    ConnectorLink,
    ProjectID,
    TableIDv2,
    TableName,
)
from statedb.config.table import TableDataVersion
from statedb.db.services import (
    ConnectorService,
    ProjectService,
    StateServices,
    TableService,
)
from statedb.rest.connector_authorization import CONNECTOR_TABLE_SOURCE_ACTIONS
from statedb.utils.errors import (
    TableDefinitionAlreadyExists,
    TableDoesNotExistError,
    TableExistsInProjectError,
)

logger = logging.getLogger(__name__)


class TableDriver:
    """
    Driver for table operations. This class exclusively performes operations
    on the v2 object models. So all operations supported here can be
    interpretted as operations supported for v2 table objects.
    """
    def __init__(
        self,
        data_backend_ctx: DataBackendContext,
        data_scale_limits: DataScaleLimits,
    ):
        self._data_backend_ctx = data_backend_ctx
        self._data_scale_limits = data_scale_limits
        self._state_services: Optional[StateServices] = None

    def _configure_state_services(self, state_services: StateServices) -> None:
        self._state_services = state_services

    def _require_state_services(self) -> StateServices:
        if self._state_services is None:
            raise RuntimeError("TableDriver StateServices is not configured")
        return self._state_services

    def _table_service(self) -> TableService:
        return self._require_state_services().tables

    def _connector_service(self) -> ConnectorService:
        return self._require_state_services().connectors

    def _project_service(self) -> ProjectService:
        return self._require_state_services().projects

    def _require_table_service(self) -> TableService:
        return self._table_service()

    def _require_connector_service(self) -> ConnectorService:
        return self._connector_service()

    def _require_project_service(self) -> ProjectService:
        return self._project_service()

    async def _require_project_access(self,
                                      project_id: ProjectID | str) -> None:
        """Authorize a project before any table repository or cache read."""
        await self._require_project_service().get_project(
            ProjectID(project_id),
            _current_user_ctx.get(),
        )

    async def _get_table_entry_by_fpid(
        self,
        project_id: ProjectID,
        table_fpid: FPID,
    ) -> Optional[TableDBEntryV2]:
        return await self._require_table_service().get_table_entry(
            project_id, table_fpid)

    async def _list_named_table_entries(
        self,
        project_id: ProjectID,
    ) -> List[TableDBEntryV2]:
        return await self._require_table_service().list_named_tables(project_id)

    async def _get_table_entry_by_id(
        self,
        project_id: ProjectID,
        table_id: TableIDv2,
    ) -> Optional[TableDBEntryV2]:
        _, _, table_fpid = TableIDv2.parse(table_id)
        assert table_fpid is not None
        return await self._require_table_service().get_table_entry(
            project_id, table_fpid)

    async def _get_table_definition_by_id(
        self,
        project_id: ProjectID,
        table_id: TableIDv2,
    ) -> Optional[TableDefinition]:
        _, _, table_fpid = TableIDv2.parse(table_id)
        assert table_fpid is not None
        return await self._require_table_service().get_table_definition(
            project_id, table_fpid)

    async def _get_table_entries_by_fpids(
        self,
        project_id: ProjectID,
        table_fpids: List[FPID],
    ) -> Dict[FPID, TableDBEntryV2]:
        return await self._require_table_service().get_table_entries(
            project_id, table_fpids)

    async def _execution_ref(
        self,
        connector_id: ConnectorID,
        project_id: str,
    ) -> ConnectorExecutionRef:
        connector_service = self._require_connector_service()
        return await connector_service.execution_ref_for_project(
            _current_user_ctx.get(),
            ProjectID(project_id),
            connector_id,
            CONNECTOR_TABLE_SOURCE_ACTIONS,
        )

    async def _table_resource_responses_from_entries(
        self,
        table_entries: List[TableDBEntryV2],
        fetch_column_stats: bool = True,
    ) -> List[TableResourceResponse]:
        table_defs = [entry.config for entry in table_entries]
        cache = get_active_cache()
        exact_stats = await cache.get_many_min_latest_exact_stats.coroutine(
            table_defs,
            fetch_column_stats,
        )

        results: List[TableResourceResponse] = []
        for table_entry, stats in zip(table_entries, exact_stats):
            table_def = table_entry.config
            connector_id = table_def.get_connector_id()
            results.append(
                TableResourceResponse(
                    id=table_entry.id,
                    table=table_def,
                    name_alias=table_entry.alias,
                    timestamps=table_entry.timestamps,
                    stats=get_table_stats(stats),
                    time_range=get_full_time_range(table_def, stats),
                    connector=ConnectorLink(connector_id, connector_id),
                    table_kind=derive_table_kind(table_def),
                    created_by=table_entry.created_by,
                    created_by_name=table_entry.created_by_name,
                ))
        return results

    async def _validate_new_table_for_state_service(
        self,
        table: TableDefinition,
        *,
        project_id: str,
        max_table_size_limits: Optional[Dict[str, int]] = None,
    ) -> None:
        if table.is_sql:
            # A Custom SQL Table has no physical source table: validate its query
            # schema in the dedicated activity rather than probing the synthetic
            # alias as a physical table (which would fail, or validate an
            # unrelated table that happens to share the synthetic name).
            connector_type = await self._validate_custom_sql_request(
                table.get_connector_id(), table.source_table.sql, project_id)
            reference_stypes = await _sql_reference_stype_hints(
                self._require_state_services(), _current_user_ctx.get(),
                project_id, table.get_connector_id(), table.source_table.sql,
                connector_type=connector_type)
            reference = await self._execution_ref(table.get_connector_id(),
                                                  project_id)
            response = await DataBackend._execute_single_activity(
                self._data_backend_ctx,
                validate_custom_sql,
                [reference,
                 _sql_resolve_request(table), reference_stypes],
                f'validate-custom-sql:{project_id}',
                execution_timeout=timedelta(minutes=5),
            )
            if response.table_definition is None:
                raise ValidationException(
                    'Custom SQL Table Validation Failure',
                    ' '.join(error.message for error in response.errors)
                    or 'The SQL query is not a valid Custom SQL Table.')
            return
        reference = await self._execution_ref(table.get_connector_id(),
                                              project_id)
        await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            validate_table_definition_activity,
            [reference, table, max_table_size_limits or {}],
            f'validate-table:{project_id}',
        )

    async def create_full_table(
        self,
        table: TableDefinition,
        *,
        project_id: str,
        alias: str,
        force_rename: bool = False,
    ) -> TableIDv2:
        """Validate and persist a table.
          A new physical table definition is validated before it is first persisted.
          Existing Custom SQL definitions are revalidated before an alias or create
          operation can reactivate them.
          Args:
              table: Complete table definition to persist.
              project_id: Project in which the table will be created.
              alias: Project-local display name.
              force_rename: Whether an existing conflicting alias may be reassigned.
          Returns:
              The public identifier of the persisted table.
          Raises:
              ValueError: If ``project_id`` or ``alias`` is missing.
              TableExistsInProjectError: If ``alias`` already exists and
                  ``force_rename`` is false.
              ValidationException: If the table definition or Custom SQL is invalid.
              SourceTableNotFoundError: If a physical source table is definitively
                  absent.
              SourceTableUnavailableError: If the provider cannot distinguish a
                  missing source from an inaccessible source.
          """
        if project_id is None:
            raise ValueError('project_id is required')
        if not alias:
            raise ValueError('alias is required')
        table = await self._canonicalize_custom_sql_name(table, project_id)
        await self._require_project_service().validate_project_for_creation(
            project_id,
            _current_user_ctx.get(),
        )

        if alias and force_rename is False:
            table_id = TableIDv2.public(
                pquery_id=None,
                table_name=TableName("NA"),
                fpid=table.fpid,
            )
            name_validation = await self.validate_project_table_name(
                project_id,
                alias,
                exclude_table_id=table_id,
            )
            if not name_validation.is_valid:
                raise TableExistsInProjectError(
                    name_validation.error_msg
                    or "Table display name already exists in project.")

        table_id = await self._persist_table_definition(
            table=table,
            alias=alias,
            force_rename=force_rename,
            project_id=project_id,
        )

        return table_id

    async def _persist_table_definition(
        self,
        table: TableDefinition,
        alias: Optional[str],
        force_rename: bool,
        project_id: str,
    ) -> TableIDv2:
        table_service = self._require_table_service()
        typed_project_id = ProjectID(project_id)
        existing_entry = await table_service.get_table_entry(
            typed_project_id, table.fpid)
        if existing_entry is None:
            await self._validate_new_table_for_state_service(
                table, project_id=project_id)
        elif table.is_sql:
            # A matching fingerprint may predate the namespace guard. Re-check
            # it before an alias/create operation can reactivate that definition.
            await self._validate_custom_sql_request(table.get_connector_id(),
                                                    table.source_table.sql,
                                                    project_id)

        try:
            principal = _current_user_ctx.get()
            if principal is None:
                raise RuntimeError('Authenticated principal is required')
            await table_service.save_table_definition(
                typed_project_id,
                table,
                alias=alias,
                force_rename=force_rename,
                increment_ref_count=False,
                created_by=str(principal.user_id),
                created_by_name=principal.user_name,
            )
        except TableDefinitionAlreadyExists:
            pass

        self._data_scale_limits.validate_table(
            table_name=TableName(alias or "NA"),  # to fix.
            num_cols=len(table.cols),
            raise_on_error=True,
            approx=True,
        )

        return TableIDv2.public(
            pquery_id=None,
            table_name=TableName("NA"),
            fpid=table.fpid,
        )

    async def delete_table_alias(
        self,
        table_alias: str,
        project_id: str,
    ) -> TableIDv2:
        """Clear a table alias without deleting its fingerprint resource."""
        await self._require_project_access(project_id)
        entry = await self._require_table_service().clear_table_alias(
            ProjectID(project_id), table_alias)
        if entry is None:
            raise TableDoesNotExistError(table_alias)
        return TableIDv2.public(
            pquery_id=None,
            table_name=TableName("NA"),
            fpid=entry.id,
        )

    async def refresh_table(
        self,
        table_id: TableIDv2,
        project_id: str,
    ):
        """Trigger an asynchronous refresh of a table's data.

        If the table exists, the method initiates an asynchronous refresh
        operation to update the table's data from its source.

        If the table does not exist, the method raises a
        :class:`TableDoesNotExistError` exception.

        Args:
            table_id: The unique identifier of the table to refresh.

        Raises:
            TableDoesNotExistError: If the table with the specified ID does not
                exist.
            DataBackendError: If there are issues with the underlying data
                backend or storage system.

        Note:
            This method only requires read access to the database to verify the
                table exists, but the refresh operation itself is performed
                asynchronously.
        """
        await self._require_project_access(project_id)
        table_entry = await self._get_table_entry_by_id(ProjectID(project_id),
                                                        table_id)
        if table_entry is None:
            raise TableDoesNotExistError(table_id)
        await self._trigger_async_refresh_for_table_entry(
            table_entry, project_id=project_id)

    async def _trigger_async_refresh_for_table_entry(
        self,
        table_entry: TableDBEntryV2,
        *,
        refresh_source: bool = True,
        project_id: str,
    ) -> TableDataVersion:
        table_def = table_entry.config
        if table_def.is_sql:
            # Gate persisted definitions before any credential or workflow is
            # dispatched; this also covers definitions saved before scope
            # enforcement was introduced.
            await self._validate_custom_sql_request(
                table_def.get_connector_id(), table_def.source_table.sql,
                project_id)
        table_alias = (TableName(table_entry.alias)
                       if table_entry.alias else None)
        connector_id = table_def.get_connector_id()
        source_table_name = str(table_def.get_source_table_name())
        connector_service = self._require_connector_service()
        execution_spec = await connector_service.execution_ref_for_project(
            _current_user_ctx.get(), ProjectID(project_id), connector_id,
            CONNECTOR_TABLE_SOURCE_ACTIONS)

        src_table_data_ver = (
            await get_active_cache().table_version_cache.get.coroutine(
                connector_id, source_table_name))

        # A Custom SQL table's data_version is a content hash over the SQL +
        # output cols + referenced-table versions; the name-keyed
        # table_version_cache cannot represent those, so a SQL table must
        # always (re)compute rather than serve a synthetic-name cache hit.
        force_source_refresh = refresh_source or src_table_data_ver is None
        if table_def.is_sql or force_source_refresh:
            try:
                src_table_data_ver = await DataBackend.get_table_data_version_in_activity(
                    self._data_backend_ctx,
                    execution_spec,
                    source_table_name,
                    refresh_table_version=force_source_refresh,
                    table_def=table_def,
                )
            except (FileNotFoundError, ConnectorError) as e:
                raise ValidationHTTPException(
                    TABLE_READ_ERROR_MESSAGE.format(source_table_name,
                                                    connector_id,
                                                    TABLE_READ_PROVIDER_ERROR),
                    e,
                )
            execution_spec, = await connector_service.fresh_execution_refs_for_workflow(
                _current_user_ctx.get(),
                (execution_spec, ),
                CONNECTOR_TABLE_SOURCE_ACTIONS,
            )

        data_ver = TableDataVersion(src_table_data_ver)
        refresh_result = (await DataBackend.get_many_table_refresh_results([
            (table_def, data_ver)
        ]))[0]
        if refresh_result:
            refresh_wf_id = TableRefreshWorkflow.compose_wf_id(
                table_def, data_ver)
            if not refresh_result.status.completed:
                tclient = self._data_backend_ctx.temporal_client()
                wf_desc = await tclient.get_workflow_desc(refresh_wf_id)
                if (wf_desc
                        and wf_desc.status == WorkflowExecutionStatus.RUNNING):
                    logger.info("Table refresh workflow is already running: %s",
                                refresh_wf_id)
                    return data_ver

            if (refresh_result.status == AsyncDataTaskStatus.DONE
                    and refresh_result.stage == TableRefreshStage.COMPLETED
                    and not refresh_source):
                logger.info("Table refresh is already completed, wf: %s",
                            refresh_wf_id)
                return data_ver

        await DataBackend.start_refresh_table(self._data_backend_ctx,
                                              execution_spec, table_def,
                                              data_ver, alias=table_alias)

        return data_ver

    async def get_table_last_ingested_timestamp(
        self,
        table_id: TableIDv2,
        project_id: str,
    ) -> Optional[datetime]:
        """Retrieve the time at which this table was last ingested.

        Args:
            table_id: The ID of the table we want information about.

        Returns:
            Optional[datetime]: Timestamp if successfully ingested, else None.
        """
        await self._require_project_access(project_id)
        table_id_v2 = TableIDv2(table_id)
        table_def = await self._get_table_definition_by_id(
            ProjectID(project_id), table_id_v2)
        if table_def is None:
            raise TableDoesNotExistError(table_id)
        return await self._get_last_ingested_timestamp_for_table_def(
            table_id,
            table_def,
        )

    async def _get_last_ingested_timestamp_for_table_def(
        self,
        table_id: TableIDv2,
        table_def: TableDefinition,
    ) -> Optional[datetime]:
        cache = get_active_cache()
        versioned_source_table_id = (
            await cache.table_version_cache.get_many.coroutine(
                tables=[table_def.source_table_id]))[0]

        if versioned_source_table_id is None:
            logger.warning(
                'No latest table data version found for table %s',
                table_id,
            )
            latest_table_data_version = None
        else:
            latest_table_data_version = versioned_source_table_id.version

        ingestion_info = (await cache.ingestion_info.get_many.coroutine(
            [(
                table_def.get_connector_id(),
                table_def.get_source_table_name(),
                latest_table_data_version,
            )],
            TableIngestionInfo,
        ))[0]

        return (ingestion_info.ingest_start_time if isinstance(
            ingestion_info, TableIngestionInfo) else None)

    async def infer_table_metadata(
        self,
        request: table_api.TableMetadataRequest,
        project_id: str,
    ) -> table_api.TableMetadataResponse:
        """Infer metadata for a table from its data source.

        If the metadata inference is successful, the method returns a
        :class:`TableMetadataResponse` object containing the inferred column
        types, statistics, and other metadata.

        If the data source is invalid or inaccessible, the method raises an
        appropriate exception.

        Args:
            request: The metadata request containing information about the
                data source to analyze.

        Returns:
            TableMetadataResponse: The inferred metadata including column
                types, statistics, and data source information.

        Raises:
            ValidationException: If the request is invalid user input, e.g.
                it references a column or source table that does not exist
                behind the connector.
            DataBackendError: If there are issues accessing the data source
                or underlying storage system.

        Note:
            Without StateServices, this method requires read-write access to
            the database for connector management operations.
        """
        source_table, _, connector_id = extract_source_table_type(
            request.source_table)
        assert connector_id is not None
        # A Custom SQL table inherits stypes from the sibling Kumo tables its
        # query reads; resolve those hints REST-side (state DB is reachable) and
        # append them as a 3rd activity arg. Physical tables keep the prior
        # 2-arg ``[reference, request]`` arity so a rolling deploy to an old
        # data-backend worker (2-param signature) still infers physical schemas.
        if isinstance(source_table, CustomSQLSourceTable):
            # Reject a non-warehouse connector before dispatching inference --
            # the same authoritative gate the create/validate paths apply,
            # resolved from credential-free connector metadata. Reuse the
            # resolved dialect for the stype hints so no second connector read is
            # needed.
            connector_type = await self._validate_custom_sql_request(
                connector_id, source_table.sql, project_id)
            reference = await self._execution_ref(connector_id, project_id)
            activity_inputs = [reference, request]
            activity_inputs.append(await _sql_reference_stype_hints(
                self._require_state_services(), _current_user_ctx.get(),
                project_id, connector_id, source_table.sql,
                connector_type=connector_type))
        else:
            reference = await self._execution_ref(connector_id, project_id)
            activity_inputs = [reference, request]
        return await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            infer_table_metadata_activity,
            activity_inputs,
            f'infer-table-metadata:{project_id}',
        )

    async def _get_many_table_configs_for_sdk_from_entries(
        self,
        project_id: ProjectID,
        table_name_to_dbid: Dict[TableName, FPID],
        db_entries: Dict[FPID, TableDBEntryV2],
    ) -> Tuple[List[ConnectorSpec], List[TableReferenceMetadata]]:
        table_fpids: Set[FPID] = set()
        table_fpid_to_name_in_graph: Dict[FPID, str] = {}
        for name, fpid in table_name_to_dbid.items():
            table_fpid_to_name_in_graph[fpid] = str(name)
            table_fpids.add(fpid)

        table_confs: List[TableReferenceMetadata] = []
        connector_confs_set = set()
        connector_spec_arr: List[ConnectorSpec] = []
        table_logic = TableLogic(
            self._data_backend_ctx,
            state_services=self._state_services,
            project_id=project_id,
        )
        for fpid, table_entry in db_entries.items():
            if fpid not in table_fpids:
                continue
            table_def = table_entry.config
            table_id = TableIDv2.public(
                pquery_id=None,
                table_name=TableName("NA"),
                fpid=fpid,
            )
            connector_conf, src_tbl_conf = (
                table_logic.get_source_table_and_connector_metadata(
                    table_entry,
                    table_id,
                    table_fpid_to_name_in_graph,
                ))

            table_metadata = {
                "cols": table_def.cols,
                "pkey": table_def.pkey,
                "time_col": table_def.time_col,
                "end_time_col": table_def.end_time_col,
            }
            table_config = TableReferenceMetadata(
                source_table_name=src_tbl_conf["table_name"],
                table_fpid=fpid,
                table_config={
                    **src_tbl_conf,
                    **table_metadata,
                },
                table_alias_in_graph=table_fpid_to_name_in_graph[fpid],
            )
            if table_entry.alias:
                table_config.table_alias = table_entry.alias

            table_confs.append(table_config)
            connector_confs_set.add(connector_conf)

        for config in connector_confs_set:
            config_as_dict = dict(config)
            if "connector_id" in config_as_dict:
                connector_id = ConnectorID(config_as_dict["connector_id"])
            elif "name" in config_as_dict:
                connector_id = ConnectorID(config_as_dict["name"])
            else:
                raise ValueError(f"Connector with config {config_as_dict} does "
                                 "not have a valied connector ID.")
            connector = await self._require_connector_service(
            ).source_table_connector_for_project(
                project_id,
                connector_id,
                _current_user_ctx.get(),
            )
            config_as_dict["connector_id"] = str(connector_id)
            if connector is None:
                config_as_dict["is_project_upload"] = True
            else:
                config_as_dict["lookup_name"] = connector.name
            connector_spec_arr.append(
                ConnectorSpec(
                    connector_id=connector_id,
                    connector_config=config_as_dict,
                ))

        return connector_spec_arr, table_confs

    async def validate_table(
        self,
        table_definition: table_api.TableDefinition,
        project_id: str,
    ) -> ValidationResponse:
        """Validate a table definition for correctness and constraints.

        If the table definition is valid, the method returns a
        :class:`ValidationResponse` object with no errors.

        If the table definition is invalid, the method returns a
        :class:`ValidationResponse` object containing validation errors.

        Args:
            table_definition: The table definition to validate, including
                columns, data source, and configuration parameters.

        Returns:
            ValidationResponse: A response object indicating whether the
                table definition is valid and containing any validation errors.

        Raises:
            DataBackendError: If there are issues with the underlying data
                backend or storage system during validation.

        Note:
            This method requires read-write access to the database for
            comprehensive validation checks.
        """
        table_def = TableDefinition(**dataclasses.asdict(table_definition))

        self._require_table_service()
        try:
            await self._validate_new_table_for_state_service(
                table_def,
                project_id=project_id,
                max_table_size_limits=(self._data_scale_limits.parse_from_env().
                                       max_table_size_bytes_by_connector),
            )
        except Exception as e:
            # TODO(manan): fix wide net:
            return ValidationResponse(errors=[
                ValidationError(
                    title="Table Validation Failure",
                    message=str(e),
                )
            ])
        return ValidationResponse()

    async def resolve_sql(
        self,
        request: table_api.SqlTableResolveRequest,
        *,
        project_id: str,
    ) -> table_api.SqlTableResolveResponse:
        """Resolve and validate a Custom SQL table."""
        connector_id = ConnectorID(request.source_table.connector_id)
        try:
            connector_type = await self._validate_custom_sql_request(
                connector_id, request.source_table.sql, project_id)
        except ValueError as error:
            return table_api.SqlTableResolveResponse(errors=[
                ApiValidationError(title='Invalid Custom SQL table',
                                   message=str(error))
            ])
        reference = await self._execution_ref(connector_id, project_id)
        reference_stypes = await _sql_reference_stype_hints(
            self._require_state_services(), _current_user_ctx.get(), project_id,
            connector_id, request.source_table.sql,
            connector_type=connector_type)
        return await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            validate_custom_sql,
            [reference, request, reference_stypes],
            f'resolve-custom-sql:{project_id}',
            execution_timeout=timedelta(minutes=5),
        )

    async def _validate_custom_sql_request(
        self,
        connector_id: ConnectorID,
        sql: str,
        project_id: str,
    ) -> ConnectorType:
        """Reject unsafe or out-of-scope SQL before any worker dispatch."""
        connector = await self._connector_service().get_for_project(
            ProjectID(project_id), connector_id, _current_user_ctx.get())
        connector_type = ConnectorType(connector.config.type.value)
        authoritative_sql_connector_type(connector_type)
        advisory_read_only_check(sql, connector_type)
        validate_connector_sql_scope(sql, connector_type, connector.config)
        return connector_type

    async def _sql_dialect(
        self,
        connector_id: ConnectorID,
        project_id: str,
    ) -> ConnectorType:
        """Warehouse dialect from credential-free connector metadata (no
        identity resolution). Raises when the connector is missing/unreadable —
        callers that must be authoritative (name canonicalization) let it
        propagate."""
        connector = await self._connector_service().get_for_project(
            ProjectID(project_id), connector_id, _current_user_ctx.get())
        return ConnectorType(connector.config.type.value)

    async def _canonicalize_custom_sql_name(
        self,
        table: TableDefinition,
        project_id: str,
    ) -> TableDefinition:
        """Assign a Custom SQL table's server-canonical synthetic name and
        record the external tables its query reads, before it is persisted.

        The synthetic ``table`` name is the identity a SQL table's fingerprint
        path-id and cache/staging keys hang off. The SDK sets a provisional
        ``md5(raw sql)`` that churns on formatting-only edits; the server
        recomputes it from the *normalized* SQL (matching the name the
        Validate-time stats are persisted under) so a whitespace/comment/case-
        only edit keeps the same path-id — the create dedupes onto the existing
        fingerprint entry and reuses its materialization/stats instead of
        orphaning them.

        ``referenced_tables`` is likewise recomputed server-side from the SQL
        (rather than trusting the client) so lineage/display read the persisted
        dependency set without re-parsing. Best-effort: a query sqlglot cannot
        parse persists an empty list rather than blocking. A no-op for physical
        tables and when neither the name nor the references change.

        The dialect is resolved authoritatively (``_sql_dialect``, not the
        advisory ``_or_none``): a SQL table whose warehouse can't be typed must
        fail creation loudly rather than persist the SDK's provisional
        ``md5(raw sql)`` name, which would silently diverge from the canonical
        fingerprint the Validate-time stats were written under.
        """
        if not table.is_sql:
            return table
        connector_type = await self._sql_dialect(table.get_connector_id(),
                                                 project_id)
        return canonicalize_custom_sql_source(table, connector_type)

    async def get_table(
        self,
        table_fpid: FPID,
        project_id: str,
    ) -> TableResourceResponse:
        """Retrieve a table by its FPID.

        If the table exists, the method returns a
        :class:`TableResourceResponse` object containing the table's
        configuration, metadata, and statistics.

        If the table does not exist, the method raises a
        :class:`KeyError` exception.

        Args:
            table_fpid: The unique FPID (Fingerprint ID) of the table to
            retrieve.

        Returns:
            TableResourceResponse: A complete table resource containing the
                table's ID, configuration, alias, timestamps, statistics,
                time range, and connector information.

        Raises:
            KeyError: If the table with the specified FPID does not exist.
            DataBackendError: If there are issues with the underlying data
                backend or storage system.

        Note:
            This method requires read access to the database and fetches
            comprehensive table information including statistics.
        """
        await self._require_project_access(project_id)
        table_entry = await self._get_table_entry_by_fpid(
            ProjectID(project_id),
            table_fpid,
        )
        if table_entry is None:
            raise KeyError(f"Table id {table_fpid} is not found")
        return (await
                self._table_resource_responses_from_entries([table_entry]))[0]

    async def get_table_by_name(
        self,
        name: ID,
        project_id: str,
    ) -> TableResourceResponse:
        """Retrieve a table by its name/alias.

        If the table exists, the method returns a
        :class:`TableResourceResponse` object containing the table's
        configuration, metadata, and statistics.

        If the table does not exist, the method raises a
        :class:`KeyError` exception.

        Args:
            name: The name/alias of the table to retrieve.

        Returns:
            TableResourceResponse: A complete table resource containing the
                table's ID, configuration, alias, timestamps, statistics,
                time range, and connector information.

        Raises:
            KeyError: If the table with the specified name does not exist.
            DataBackendError: If there are issues with the underlying data
                backend or storage system.

        Note:
            This method requires read access to the database and fetches
            comprehensive table information including statistics.
        """
        await self._require_project_access(project_id)
        table_entry = await self._require_table_service(
        ).get_table_entry_by_alias(ProjectID(project_id), name)
        if table_entry is None:
            raise KeyError(f"Table name {name} is not found")
        return (await
                self._table_resource_responses_from_entries([table_entry]))[0]

    async def resolve_project_table(
        self,
        project_id: str,
        table_id_or_name: str,
    ) -> TableDBEntryV2:
        return await self._resolve_project_table(project_id, table_id_or_name)

    async def _resolve_project_table(
        self,
        project_id: str,
        table_id_or_name: str,
    ) -> TableDBEntryV2:
        """Resolve a canonical table ID or project-local alias."""
        project_id = ProjectID(project_id)
        await self._require_project_access(project_id)
        try:
            table_id = TableIDv2.from_public(table_id_or_name)
            _, _, fpid = TableIDv2.parse(table_id)
            assert fpid is not None
            table_entry = await self._require_table_service().get_table_entry(
                project_id, fpid)
        except ValueError:
            table_entry = await self._require_table_service(
            ).get_table_entry_by_alias(project_id, table_id_or_name)
        if not table_entry:
            raise TableDoesNotExistError(table_id_or_name)
        return table_entry

    async def list_project_table_names(
        self,
        project_id: str,
    ) -> List[ListResponse]:
        return await self._list_project_table_names(project_id)

    async def _list_project_table_names(
        self,
        project_id: str,
    ) -> List[ListResponse]:
        """List table names scoped to a project using project-local aliases."""
        project_id = ProjectID(project_id)
        await self._require_project_access(project_id)
        entries = await self._require_table_service().list_named_tables(
            project_id)
        return [
            ListResponse(
                name=entry.alias,
                id=TableIDv2.public(
                    pquery_id=None,
                    table_name=TableName("NA"),
                    fpid=entry.id,
                ),
                connector_id=entry.config.get_connector_id(),
                source_table_name=str(entry.config.get_source_table_name()),
                timestamps=entry.timestamps,
                table_kind=derive_table_kind(entry.config),
            ) for entry in entries if entry.alias
        ]

    async def list_table_names(
        self,
        project_id: str,
    ) -> List[ListResponse]:
        """Retrieve a list of all table names in the system.

        The method returns a list of :class:`ListResponse` objects containing
        the names and basic information of all tables in the database, except
        the ones created from the SDK and not saved with a name.

        Returns:
            List[ListResponse]: A list of table name responses containing
                the names and basic metadata of all tables.

        Raises:
            DataBackendError: If there are issues with the underlying data
                backend or storage system.

        Note:
            This method requires read access to the database and returns
            lightweight information about all tables.
        """
        await self._require_project_access(project_id)
        table_entries = await self._list_named_table_entries(
            ProjectID(project_id))
        return [
            ListResponse(
                name=entry.alias,
                id=TableIDv2.compose(
                    None,
                    TableName("NA"),
                    FPID(entry.id),
                ),
                connector_id=entry.config.get_connector_id(),
                source_table_name=str(entry.config.get_source_table_name()),
                timestamps=entry.timestamps,
            ) for entry in table_entries if entry.alias
        ]

    async def get_col_stats_and_sample_rows(
        self,
        id: TableIDv2,
        project_id: str,
    ) -> ColStatsResponse:
        """Retrieve column statistics and sample rows for a table.

        If the table exists, the method returns a
        :class:`ColStatsResponse` object containing column statistics
        and sample data rows.

        If the table does not exist, the method raises a
        :class:`TableDoesNotExistError` exception.

        Args:
            id: The unique identifier of the table to analyze.

        Returns:
            ColStatsResponse: Column statistics and sample rows containing
                data type information, statistics, and sample data.

        Raises:
            TableDoesNotExistError: If the table with the specified ID does not
                exist.
            DataBackendError: If there are issues with the underlying data
                backend or storage system.

        Note:
            This method requires read access to the database and performs
            data analysis to generate statistics and sample rows.
        """
        await self._require_project_access(project_id)
        _, _, table_fpid = TableIDv2.parse(id)
        if table_fpid is None:
            raise TableDoesNotExistError(id)
        table_def = await self._get_table_definition_by_id(
            ProjectID(project_id), id)
        if table_def is None:
            raise TableDoesNotExistError(id)
        reference = await self._execution_ref(table_def.get_connector_id(),
                                              project_id)
        return await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            get_table_col_stats_and_sample_rows_activity,
            [reference, table_def],
            f'table-stats:{project_id}',
        )

    async def list_tables(
        self,
        project_id: str,
        connector_id: Optional[ConnectorID] = None,
        fetch_column_stats: bool = True,
    ) -> List[TableResourceResponse]:
        """Retrieve a list of tables with optional filtering and statistics.

        If tables exist, the method returns a list of
        :class:`TableResourceResponse` objects containing complete
        table information and optionally column statistics.

        Args:
            connector_id: Optional connector ID to filter tables by data
                source. If None, returns all tables.
            fetch_column_stats: If True, includes column statistics in the
                response. If False, returns basic table information without
                statistics.

        Returns:
            List[TableResourceResponse]: A list of complete table resources
                containing table configurations, metadata, and optionally
                column statistics.

        Raises:
            DataBackendError: If there are issues with the underlying data
                backend or storage system.

        Note:
            This method requires read access to the database. Setting
            :param:`fetch_column_stats` to False can improve performance
            when statistics are not needed.
        """
        await self._require_project_access(project_id)
        table_entries = await self._require_table_service().list_tables(
            ProjectID(project_id), connector_id=connector_id)
        return await self._table_resource_responses_from_entries(
            table_entries,
            fetch_column_stats=fetch_column_stats,
        )

    async def validate_table_name(
        self,
        name: str,
        project_id: str,
        exclude_table_id: Optional[str] = None,
    ) -> ValidateNameResponse:
        """Validate a table name for correctness and availability.

        If the table name is valid and available, the method returns a
        :class:`ValidateNameResponse` object indicating success.

        If the table name is invalid or already taken, the method returns a
        :class:`ValidateNameResponse` object containing validation errors.

        Args:
            name: The table name to validate for correctness and availability.
            project_id: Optional project ID to scope validation to. When
                provided, validates the name within the
                project scope.
            exclude_table_id: Optional table ID to exclude from the uniqueness
                check (used when renaming an existing table).

        Returns:
            ValidateNameResponse: A response object indicating whether the
                table name is valid and available for use.

        Raises:
            DataBackendError: If there are issues with the underlying data
                backend or storage system during validation.

        Note:
            This method requires read access to the database to check
            name availability and validate naming conventions.
        """
        return await self._validate_project_table_name(
            project_id=project_id,
            name=name,
            exclude_table_id=exclude_table_id,
        )

    async def validate_project_table_name(
        self,
        project_id: str,
        name: str,
        exclude_table_id: Optional[str] = None,
    ) -> ValidateNameResponse:
        return await self._validate_project_table_name(
            project_id,
            name,
            exclude_table_id=exclude_table_id,
        )

    async def _validate_project_table_name(
        self,
        project_id: str,
        name: str,
        exclude_table_id: Optional[str] = None,
    ) -> ValidateNameResponse:
        """Validate that a project-local table display name is available."""
        try:
            TableName(name)
        except ValueError as e:
            return ValidateNameResponse(is_valid=False, error_msg=str(e))

        typed_project_id = ProjectID(project_id)
        await self._require_project_access(typed_project_id)
        existing = await self._require_table_service().get_table_entry_by_alias(
            typed_project_id, name)
        if existing is None:
            return ValidateNameResponse(is_valid=True, error_msg=None)
        if exclude_table_id is not None:
            try:
                _, _, excluded_fpid = TableIDv2.parse(
                    TableIDv2.from_public(exclude_table_id))
            except ValueError:
                excluded_fpid = None
            if excluded_fpid == existing.id:
                return ValidateNameResponse(is_valid=True, error_msg=None)
        return ValidateNameResponse(
            is_valid=False,
            error_msg=f"Table with name '{name}' already exists in the project.",
        )

    async def table_to_sdk(
        self,
        table_id: TableIDv2,
        project_id: str,
    ) -> str:
        """
        Used to generate a fully executable Python SDK script. The script contains:
            1. Complete table config structured as Python dictionary.
            2. Connector setup.
            3. Load the desired table in the sdk via `kumo.Table.from_source_table()`
            using the metadata provided in the config.
        This method is part of the UI to SDK feature and will be called in the UI.

        Args:
            table_id (TableID): The table ID for the table that we will be
            generating the config and script.
        Response:
            str: The complete python script as a string.
        """
        await self._require_project_access(project_id)
        _, table_name, fpid = TableIDv2.parse(table_id)
        if fpid is None:
            raise ValueError(
                f"Unsupported table ID: {table_id}." +
                " Please pass a table ID with the format: pquery_id.table_name-fpid."
            )
        configs = await self._get_many_table_configs_for_sdk_from_entries(
            ProjectID(project_id),
            {table_name: fpid},
            await self._get_table_entries_by_fpids(ProjectID(project_id),
                                                   [fpid]),
        )
        connector_configs, table_configs = configs

        return make_stype_and_dtype_sdk_compat(
            generate_conn_graph_or_table_sdk_script(connector_configs,
                                                    table_configs,
                                                    project_id=project_id))

    async def get_table_validation(
        self,
        table_id: str,
        project_id: str,
        data_version: Optional[str] = None,
    ) -> Optional[TableDataTaskResult]:
        """
        Fetch the table validation after a table has been
        ingested. If the table has not been ingested yet
        or doesn't contain the validation response,
        return None.

        Args:
            table_id [str]: The id of the table of interest in the
            format NA.NA-<fpid>.

        Returns:
            TableDataTaskResult if present. None otherwise.
        """
        await self._require_project_access(project_id)
        table_id = TableIDv2(table_id)
        table_entry = await self._get_table_entry_by_id(ProjectID(project_id),
                                                        table_id)

        if not table_entry:
            raise TableDoesNotExistError(table_id)

        table_def: TableDefinition = table_entry.config

        if not data_version:
            # Check if source table already has successfully ingested data_version
            src_table_data_ver = (
                await get_active_cache().table_version_cache.get.coroutine(
                    table_def.get_connector_id(),
                    table_def.get_source_table_name()))

            # Refresh the source table's data version if needed
            if src_table_data_ver is None:
                return None
        else:
            src_table_data_ver = data_version

        data_ver = TableDataVersion(src_table_data_ver)
        refresh_result = (await DataBackend.get_many_table_refresh_results([
            (table_def, data_ver)
        ]))[0]

        return refresh_result
