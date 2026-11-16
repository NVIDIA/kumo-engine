# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os.path as osp
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import kumoapi.table as table_api
import pandas as pd
import pyarrow.parquet as pq
from deprecated import deprecated
from kumo_core.training.options import allow_array_targets
from kumo_core.types import ColStatType, ConnectorType, PredictionArtifactType
from kumo_core.types.conversions import can_cast_safely
from kumo_core.types.schema import FPID, TS_FORMAT_METADATA, BaseColumnConfig
from kumoapi.common import ValidationError as ApiValidationError
from kumoapi.table import TimestampUnit
from kumoapi.typing import Dtype, Stype
from temporalio.client import WorkflowExecutionStatus

from async_utils.instrument import measure_blocking_code
from async_utils.sync import blocking_func
from common_utils import string
from common_utils.error import PublicException
from io_utils import fsutil, logging
from io_utils.json import load_json
from kumo.config.graph import TableMetadataConfig
from kumo.connector.connector import ConnectorError, DuplicateColumnsError
from kumo.connector.sql_connector import SQLConnector
from kumo.data import get_active_cache
from kumo.data.backend_cache import (
    AsyncDataTaskStatus,
    TableIngestionInfo,
    TableRefreshStage,
    col_stat_key,
)
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from kumo.data.custom_sql import (
    CustomSqlOutputSchemaError,
    UnsupportedColumnTypeError,
    advisory_read_only_check,
    custom_sql_stats_version,
    custom_sql_synthetic_name,
    normalized_sql_identifier,
    normalized_sql_identifier_leaf,
    referenced_tables_or_none,
    sqlglot_dialect,
    subquery_safe_sql,
    validate_connector_sql_scope,
    validate_sql_output_contract,
)
from kumo.data.utils import get_diskgraph_local_ingest_path
from kumo.graph import Column as BackendColumn
from kumo.graph.graph import Connector, Table
from kumo.home import HomeDirManager
from kumo.stats import NUM_ROWS_FOR_APPROX_STATS
from kumo.stats.stats import Stats
from kumo.stats.utils import col_stat_dict_complete
from kumo.utils import format_bytes
from kumo.validation import ValidationException
from rest.backends.data_backend import connector_activity_registry
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.data_backend import DataBackend
from rest.backends.data_backend.utils import (
    _connector_type_from_spec,
    get_table_data_version,
    resolve_connector_and_type_async,
)
from rest.backends.data_backend.workflows import TableRefreshWorkflow
from rest.backends.ml_backend.prediction_artifacts import (
    resolve_prediction_artifact,
)
from rest.config.data_backend import (
    ConnectorExecutionRef,
    ConnectorOpenSpec,
    ConnectorSpec,
)
from rest.config.table import (
    PKEY_IS_ID,
    PKEY_PRESENT_IN_COLS,
    TIME_COL_NEQ_PKEY,
    TIME_COL_PRESENT_IN_COLS,
    ColStatsAtVersion,
    ColStatsResponse,
    ColumnStats,
    ColumnSummary,
)
from rest.config.table_v2 import (
    ADLSSourceTable,
    Column,
    CustomSQLSourceTable,
    DataSourceType,
    GCSSourceTable,
    S3SourceTable,
    SqlReferenceStypes,
    TableDBEntryV2,
    TableDefinition,
    TableReferenceMetadata,
    TableResourceResponse,
    TimestampUnit,
    _coerce_dtype,
    _coerce_stype,
    derive_table_kind,
    table_args_to_kumo_table,
)
from rest.deps.context import _current_user_ctx
from rest.driver.error_constants import (
    DUPLICATE_COLUMN_NAMES_TITLE,
    INVALID_COLUMN_DATA_TYPE_TITLE,
    INVALID_MULTICATEGORICAL_COLUMN_TITLE,
    INVALID_PRIMARY_KEY_TITLE,
    INVALID_TIME_COLUMN_TITLE,
    MISSING_COLUMN_DATA_TYPE_TITLE,
    SOURCE_TABLE_NOT_FOUND_TITLE,
    TABLE_COLUMN_NOT_FOUND_TITLE,
    TABLE_READ_ERROR_MESSAGE,
    TABLE_READ_PROVIDER_ERROR,
    TABLE_SIZE_EXCEEDS_LIMIT_MESSAGE,
    TABLE_SIZE_EXCEEDS_LIMIT_TITLE,
)
from rest.utils.connector import (
    DEFAULT_NUM_SAMPLE_ROWS,
    get_sample_rows_raw,
    stringify_sample_dataframe,
)
from rest.utils.driver_common import (
    TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME,
    async_connector_lifecycle,
    extract_source_table_type,
    get_connector_from_spec,
    get_connector_from_spec_async,
)
from rest.utils.routers import ValidateNameResponse, ValidationHTTPException
from rest.utils.table import ColStatsSet, get_full_time_range, get_table_stats
from statedb.config.common import (
    ConnectorID,
    ConnectorLink,
    PredictiveQueryID,
    ProjectID,
    TableID,
    TableIDv2,
    TableName,
)
from statedb.config.table import TableDataVersion
from statedb.db.services import StateServices, TableService
from statedb.rest.connector_authorization import CONNECTOR_TABLE_SOURCE_ACTIONS
from statedb.utils.errors import (
    CustomSqlQueryError,
    TableDoesNotExistError,
    TableInvalidColumnNameError,
    UnsupportedInDiskgraphError,
    sql_query_error_message,
)

log = logging.getLogger(__name__)

_FILE_SOURCE_TABLE_TYPES = (S3SourceTable, GCSSourceTable, ADLSSourceTable)
_FILE_SOURCE_TYPES = (DataSourceType.S3, DataSourceType.GCS,
                      DataSourceType.ADLS)
ColumnMetadataDict = Dict[str, Dict[str, Union[str, Dtype, Stype, None]]]


async def _resolve_table_metadata_with_connector(
    request: table_api.TableMetadataRequest,
    connector: Connector,
    connector_type: Optional[ConnectorType] = None,
    reference_stypes: Optional[Dict[str, Dict[str, Stype]]] = None,
    discover_sql_columns: bool = True,
) -> Tuple[table_api.TableMetadataResponse, Optional[TableMetadataConfig],
           Table]:
    source_table, source_table_name, _ = extract_source_table_type(
        request.source_table)
    is_sql = isinstance(source_table, CustomSQLSourceTable)
    if is_sql and connector_type is not None:
        validate_sql_output_contract(source_table.sql, connector_type)

    cols = []
    for col in request.cols:
        st = Stype(col.stype) if col.stype is not None else None
        dt = Dtype(col.dtype) if col.dtype is not None else None
        # A delimiter is an explicit semantic hint, not merely formatting
        # metadata. Resolve omitted request types before backend inference so
        # its uniqueness heuristic cannot reinterpret the column as a scalar
        # categorical and discard the caller's proven split contract.
        if col.multicategorical_delimiter is not None:
            if st is None:
                st = Stype.multicategorical
            if dt is None:
                dt = Dtype.string
        if not allow_array_targets() and dt == Dtype.stringlist.value:
            dt = Dtype.string
        c = BackendColumn(
            col.name,
            st,
            dt,
            multicategorical_delimiter=col.multicategorical_delimiter,
        )
        if col.timestamp_format is not None:
            if isinstance(col.timestamp_format, table_api.TimestampUnit):
                c.timestamp_unit = str(col.timestamp_format)
            else:
                c.metadata = {TS_FORMAT_METADATA: str(col.timestamp_format)}
        cols.append(c)

    table = Table(
        table_name=TableName("NA"),
        data_source="",
        pkey=request.pkey,
        time_col=request.time_col,
        end_time_col=request.end_time_col,
        source_name=source_table_name,
        cols=cols,
        data_version=None,
        custom_sql=source_table.sql if is_sql else None,
    )
    table.connector = connector
    if is_sql and reference_stypes and connector_type is not None:
        table.set_sql_reference_stypes(connector_type, reference_stypes)

    log.info(f"Validating table {table} is properly specified for metadata "
             f"inferral request")

    # A Custom SQL Table's computed/aliased output columns have no physical
    # table to resolve against; its schema comes from describing the query, so
    # skip the physical-column-existence check.
    if not is_sql:
        source_table_metadata = await _get_source_table_metadata.coroutine(
            source_table_name, connector)
        source_col_names = [x.name for x in source_table_metadata.all_cols()]
        for col_name in table.data_col_dict:
            if col_name not in source_col_names:
                raise ValidationException(
                    title=TABLE_COLUMN_NOT_FOUND_TITLE,
                    message=f"Column {col_name} does not exist in table "
                    f"{source_table_name} behind connector {connector}. "
                    f"Please check your source data and try again.")

    log.info(f"Inferring metadata for table {table}")

    try:
        include_cols = {col.name for col in request.cols} or None
        if is_sql and discover_sql_columns:
            include_cols = None
        elif is_sql and include_cols is None:
            raise ValueError(
                "Explicit SQL column selection requires at least one column.")
        await table._infer_metadata_blocking.coroutine(include_cols=include_cols
                                                       )
        await table._infer_dtypes_from_stypes_blocking.coroutine()
    except FileNotFoundError as e:
        raise ValidationException(
            title=SOURCE_TABLE_NOT_FOUND_TITLE,
            message='Specified source table could not '
            f'be found in your connector: {e}') from e

    unresolved_multicats = [
        col for col in table.cols
        if (col.stype == Stype.multicategorical and col.dtype == Dtype.string
            and col.multicategorical_delimiter is None)
    ]
    if unresolved_multicats:
        names = ', '.join(repr(col.name) for col in unresolved_multicats)
        raise ValidationException(
            title=INVALID_MULTICATEGORICAL_COLUMN_TITLE,
            message=f"Columns {names} are declared multicategorical, but "
            "the existing approximate-stats sample did not infer a supported "
            "delimiter. Configure a comma/pipe/semicolon/tab delimiter or use "
            "categorical for one token per row.")
    log.info(f"Table with inferred metadata has representation {table}")

    res_cols: List[Column] = []
    for col in table.data_cols:
        ts_format = None
        if col.timestamp_unit is not None:
            ts_format = TimestampUnit(col.timestamp_unit)
        elif TS_FORMAT_METADATA in col.metadata:
            ts_format = col.metadata[TS_FORMAT_METADATA]
        res_cols.append(
            Column(
                col.name,
                col.stype,
                col.dtype,
                ts_format,
                multicategorical_delimiter=(col.multicategorical_delimiter),
            ))

    response_source = request.source_table
    if is_sql and connector_type is not None:
        response_source = dataclasses.replace(
            source_table,
            table=TableName(
                custom_sql_synthetic_name(source_table.sql, connector_type)),
            referenced_tables=(referenced_tables_or_none(
                source_table.sql, connector_type) or []),
            connector_type=authoritative_sql_connector_type(connector_type),
        )
    response = table_api.TableMetadataResponse(
        cols=res_cols,
        source_table=response_source,
        pkey=table.pkey_name,
        time_col=table.time_col_name,
        end_time_col=table.end_time_col_name,
    )
    return response, table._sql_source_metadata, table


class TableLogic:
    def __init__(
        self,
        data_backend_ctx: DataBackendContext,
        *,
        state_services: Optional[StateServices] = None,
        project_id: ProjectID,
    ):
        self._data_backend_ctx = data_backend_ctx
        self._state_services = state_services
        self._project_id = project_id

    @property
    def _table_service(self):
        return self._require_state_services().tables

    def _require_state_services(self) -> StateServices:
        if self._state_services is None:
            raise RuntimeError("TableLogic requires StateServices")
        return self._state_services

    def _require_table_service(self):
        return self._table_service

    async def _execution_connector(
        self,
        connector_id: ConnectorID,
    ) -> ConnectorExecutionRef:
        return await self._require_state_services(
        ).connectors.execution_ref_for_project(_current_user_ctx.get(),
                                               self._project_id, connector_id,
                                               CONNECTOR_TABLE_SOURCE_ACTIONS)

    async def _canonicalize_custom_sql_name(
        self,
        table_def: TableDefinition,
    ) -> TableDefinition:
        """Stamp a Custom SQL table's server-owned fields (derived
        ``connector_type``, canonical synthetic name, and server-parsed
        ``referenced_tables``) before its fingerprint is taken, mirroring
        :meth:`TableDriver.create_full_table`'s guard.

        ``connector_type`` is resolved authoritatively from credential-free
        connector metadata (``get_for_project`` -> ``config.type.value``) --
        the same resolution the infer/validate paths in this class already use
        -- so a non-warehouse connector is rejected and a formatting-only edit
        keeps the same fingerprint. A no-op for physical tables.
        """
        if not table_def.is_sql:
            return table_def
        connector = await self._require_state_services(
        ).connectors.get_for_project(self._project_id,
                                     table_def.get_connector_id(),
                                     _current_user_ctx.get())
        connector_type = ConnectorType(connector.config.type.value)
        return canonicalize_custom_sql_source(table_def, connector_type)

    async def create_table_v2(
        self,
        table_def: TableDefinition,
        alias: Optional[str] = None,
        force_rename: bool = False,
    ) -> None:
        r"""Creates a table in the V2 object model as a normalized (deduped,
        keyed by fingerprint) entry with a reference count."""
        if (isinstance(table_def.source_table, _FILE_SOURCE_TABLE_TYPES)
                and table_def.source_table.connector_id is None):
            raise ConnectorError(
                "File source tables require a project-owned connector_id.")

        # Stamp server-owned Custom SQL fields before the fingerprint is taken,
        # so this persist path cannot store a SQL table with an underived
        # `connector_type`, stale `referenced_tables`, or a provisional name.
        table_def = await self._canonicalize_custom_sql_name(table_def)

        # NOTE `fpid` is a cached property and may perform some computation:
        fpid = table_def.fpid
        table_service = self._require_table_service()
        table_entry = await table_service.get_table_entry(
            self._project_id, fpid)
        if table_entry is None:
            await self.validate_table(table_def)
        await table_service.save_table_definition(
            self._project_id,
            table_def,
            alias=alias,
            force_rename=force_rename,
        )

    async def delete_table_alias(self, alias: TableName) -> TableIDv2:
        table_entry = await self._require_table_service().clear_table_alias(
            self._project_id, alias)
        log.info(f'Deleting table alias {alias=}, '
                 f'entry exists={table_entry is not None}, '
                 f'fingerprint={table_entry.id if table_entry else None}')
        if table_entry is not None:
            return table_entry.id
        raise ValueError(f"Table {alias=} does not exist.")

    async def get_table_entry(
        self,
        id: TableIDv2,
    ) -> Optional[TableDBEntryV2]:
        """
        Get the DB entry for a table.
        """
        assert isinstance(id, TableIDv2)
        entries = await self._get_many_table_entries_v2_from_state_services(
            [id])
        return entries[0]

    async def get_table_v2(
        self,
        id: TableIDv2,
    ) -> Optional[TableDefinition]:

        assert isinstance(id, TableIDv2)
        table_entry_v2 = await self.get_table_entry(id)
        if table_entry_v2 is None:
            return None
        assert isinstance(table_entry_v2, TableDBEntryV2)
        return table_entry_v2.config

    async def get_many_last_ingested_timestamp(
        self,
        ids: List[TableID],
    ) -> List[Optional[datetime]]:
        """Get the timestamp of the last successful ingestion of each table.

        Args:
            ids (List[TableIDv2]): Table to get the latest timestamp for

        Returns:
            List[Optional[datetime]]: If None, table has not been successfully
                ingested. Otherwise, last successful ingestion time.
        """
        raw_defs = await self.get_many_table_definitions(
            [TableIDv2(id) for id in ids])
        table_defs: List[TableDefinition] = []
        for i, table_def in enumerate(raw_defs):
            if isinstance(table_def, Exception):
                raise TableDoesNotExistError(ids[i])
            table_defs.append(table_def)

        # Get the latest table data version
        cache = get_active_cache()
        versioned_source_table_ids = await cache.table_version_cache.get_many.\
            coroutine(
                tables=[td.source_table_id for td in table_defs])

        for i, ltdv in enumerate(versioned_source_table_ids):
            if ltdv is None:
                log.warning('No latest table data version found for table %s',
                            ids[i])

        latest_table_data_versions = [
            vstid.version if vstid is not None else None
            for vstid in versioned_source_table_ids
        ]

        # Get the table refresh result corresponding to the latest table data
        # version
        table_ingestion_infos = await cache.ingestion_info.get_many.coroutine(
            [(td.get_connector_id(), td.get_source_table_name(),
              latest_table_data_versions[i])
             for i, td in enumerate(table_defs)], TableIngestionInfo)

        return [
            ingestion_info.ingest_start_time if isinstance(
                ingestion_info, TableIngestionInfo) else None
            for ingestion_info in table_ingestion_infos
        ]

    async def _get_many_table_response_v2(
        self,
        tables_db_entries: List[TableDBEntryV2],
        fetch_column_stats: bool = True,
    ) -> List[TableResourceResponse]:
        """
        Get many table responses from an array of TableDBEntryV2.

        Args:
            tables_db_entries: List[TableDBEntryV2],
            fetch_column_stats: bool
        Returns:
            List[TableResourceResponse]
        """
        table_defs = [entry.config for entry in tables_db_entries]

        cache = get_active_cache()
        exact_stats = await cache.get_many_min_latest_exact_stats.coroutine(
            table_defs,
            fetch_column_stats,
        )

        results: List[TableResourceResponse] = []

        for table_def, table_db_entry, stats in zip(table_defs,
                                                    tables_db_entries,
                                                    exact_stats):
            time_range = get_full_time_range(table_def, stats)
            table_stats = get_table_stats(stats)
            connector_id = table_def.get_connector_id()

            result = TableResourceResponse(
                id=table_db_entry.id,
                table=table_db_entry.config,
                name_alias=table_db_entry.alias,
                timestamps=table_db_entry.timestamps,
                stats=table_stats,
                time_range=time_range,
                connector=ConnectorLink(connector_id, connector_id),
                table_kind=derive_table_kind(table_def),
            )
            results.append(result)

        return results

    async def _get_many_table_entries_v2_from_state_services(
        self,
        ids: List[TableIDv2],
    ) -> List[Optional[TableDBEntryV2]]:
        table_service = self._require_table_service()

        tid_to_fpid: Dict[TableIDv2, FPID] = {}
        missing_tids: List[TableIDv2] = []
        missing_pquery_table_ids: List[Tuple[TableIDv2, TableName,
                                             PredictiveQueryID]] = []
        for tid in ids:
            pqid, tname, fpid = TableIDv2.parse(tid)
            if fpid is not None:
                tid_to_fpid[tid] = fpid
            else:
                missing_tids.append(tid)
                if pqid is not None:
                    missing_pquery_table_ids.append((tid, tname, pqid))

        if missing_pquery_table_ids:
            pquery_tid_to_fpid = await self._resolve_pquery_table_fpids(
                missing_pquery_table_ids)
            tid_to_fpid.update(pquery_tid_to_fpid)
            resolved_pquery_tids = set(pquery_tid_to_fpid)
            missing_tids = [
                tid for tid in missing_tids if tid not in resolved_pquery_tids
            ]

        result_dict: Dict[FPID, TableDBEntryV2] = {}
        if missing_tids:
            tid_to_tname = {
                tid: TableIDv2.parse(tid)[1]
                for tid in missing_tids
            }
            table_entries_by_alias = (
                await table_service.get_table_entries_by_aliases(
                    self._project_id, tid_to_tname.values()))
            for tid, tname in tid_to_tname.items():
                table_entry = table_entries_by_alias.get(tname)
                if table_entry is not None:
                    result_dict[table_entry.id] = table_entry
                    tid_to_fpid[tid] = table_entry.id

        fpids = set(tid_to_fpid.values())
        if fpids:
            result_dict.update(await table_service.get_table_entries(
                self._project_id, fpids))

        result: List[Optional[TableDBEntryV2]] = []
        for tid in ids:
            fpid = tid_to_fpid.get(tid)
            if fpid is None:
                result.append(None)
                continue
            result.append(result_dict.get(fpid))

        assert len(result) == len(ids)
        return result

    async def _resolve_pquery_table_fpids(
        self,
        table_ids: List[Tuple[TableIDv2, TableName, PredictiveQueryID]],
    ) -> Dict[TableIDv2, FPID]:
        pquery_ids = list(dict.fromkeys(pqid for _, _, pqid in table_ids))
        pquery_defs_by_id = await self._require_state_services(
        ).pqueries.get_definitions_by_ids(pquery_ids)
        if not pquery_defs_by_id:
            return {}

        graph_entries_by_fpid = await self._require_state_services(
        ).graphs.get_graph_entries(self._project_id, {
            pquery_def.graph_fpid
            for pquery_def in pquery_defs_by_id.values()
        })

        resolved: Dict[TableIDv2, FPID] = {}
        for tid, tname, pqid in table_ids:
            pquery_def = pquery_defs_by_id.get(pqid)
            if pquery_def is None:
                continue
            graph_entry = graph_entries_by_fpid.get(pquery_def.graph_fpid)
            if graph_entry is None:
                continue
            table_fpid = graph_entry.table_name_to_dbid.get(tname)
            if table_fpid is not None:
                resolved[tid] = table_fpid
        return resolved

    @measure_blocking_code
    async def list_tables_v2(
        self,
        connector_id: Optional[ConnectorID] = None,
        fetch_column_stats: bool = True,
    ) -> List[TableResourceResponse]:
        tables = await self._list_table_entries_v2(connector_id=connector_id)
        return await self._get_many_table_response_v2(
            tables,
            fetch_column_stats,
        )

    async def _list_table_entries_v2(
        self,
        connector_id: Optional[ConnectorID] = None,
    ) -> List[TableDBEntryV2]:
        return await self._require_table_service().list_tables(
            self._project_id, connector_id=connector_id)

    @measure_blocking_code
    async def validate_v2_table_name(
        self,
        name: str,
    ) -> ValidateNameResponse:
        try:
            TableName(name, "Table")
        except ValueError as e:
            return ValidateNameResponse(is_valid=False, error_msg=str(e))

        if await self._require_table_service().table_alias_exists(
                self._project_id, name):
            return ValidateNameResponse(
                is_valid=False,
                error_msg=(
                    f"Table with name {name} already exists. Please choose a "
                    "different name."),
            )
        return ValidateNameResponse(is_valid=True, error_msg=None)

    @measure_blocking_code
    async def get_many_table_definitions(
        self,
        ids: List[TableIDv2],
    ) -> List[Union[TableDefinition, Exception]]:
        r"""V2 batch fetch returning `TableDefinition`s — read straight
        from ``entry.config``.
        """
        entries = await self._get_many_table_entries_v2_from_state_services(ids)
        result: List[Union[TableDefinition, Exception]] = []
        for id, entry in zip(ids, entries):
            if entry is None:
                result.append(TableDoesNotExistError(id))
                continue
            result.append(entry.config)
        return result

    @measure_blocking_code
    async def trigger_async_refresh_v2(
            self, table_def: TableDefinition, table_id: TableIDv2, *,
            refresh_source: bool = True) -> TableDataVersion:
        """v2-native entry: kick off table data ingestion asynchronously
        from a `TableDefinition` directly — no v1 round-trip, no view
        branch (v2 doesn't model views).

        ``table_id`` is the v2 id; used only to look up the user-facing
        alias on ``TableDBEntryV2`` for error messages.
        """
        table_alias: Optional[TableName] = None
        entry_v2 = await self.get_table_entry(table_id)
        if entry_v2 is not None and entry_v2.alias:
            table_alias = TableName(entry_v2.alias)
        return await self._do_async_refresh(table_def, table_alias=table_alias,
                                            refresh_source=refresh_source)

    @measure_blocking_code
    async def _do_async_refresh(
            self, table_def: TableDefinition, *,
            table_alias: Optional[TableName],
            refresh_source: bool = True) -> TableDataVersion:
        log.debug(
            "Trigger async table refresh (v2): alias=%s, refresh_source: %s",
            table_alias, refresh_source)
        connector_id = table_def.get_connector_id()
        source_table_name = str(table_def.get_source_table_name())
        execution_spec = await self._execution_connector(connector_id)

        # Check if source table already has successfully ingested data_version.
        src_table_data_ver = (
            await get_active_cache().table_version_cache.get.coroutine(
                connector_id, source_table_name))

        # Refresh the source table's data version if needed. A Custom SQL
        # table's data_version is a content hash over the SQL + output cols +
        # referenced-table versions; the name-keyed table_version_cache is
        # keyed only by the synthetic source name and cannot represent those,
        # so a SQL table must always (re)compute rather than serve a cache hit.
        force_source_refresh = refresh_source or src_table_data_ver is None
        if table_def.is_sql or force_source_refresh:
            try:
                src_table_data_ver = await DataBackend.get_table_data_version_in_activity(
                    self._data_backend_ctx, execution_spec, source_table_name,
                    refresh_table_version=force_source_refresh,
                    table_def=table_def)
            except (FileNotFoundError, ConnectorError) as e:
                raise _table_read_error_v2(table_def, e)
            execution_spec, = await self._require_state_services(
            ).connectors.fresh_execution_refs_for_workflow(
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
                # Make sure the TableRefreshWorkflow is still running. There
                # are corner cases in which workflow terminated but task result
                # didn't get updated accordingly.
                tclient = self._data_backend_ctx.temporal_client()
                wf_desc = await tclient.get_workflow_desc(refresh_wf_id)
                if (wf_desc
                        and wf_desc.status == WorkflowExecutionStatus.RUNNING):
                    log.info('Table refresh workflow is already running: %s',
                             refresh_wf_id)
                    return data_ver

            # Table refresh workflow has already completed successfully. We
            # should not relaunch the workflow unless forcing refresh_source.
            if (refresh_result.status == AsyncDataTaskStatus.DONE
                    and refresh_result.stage == TableRefreshStage.COMPLETED
                    and not refresh_source):
                log.info('Table refresh is already completed, wf: %s',
                         refresh_wf_id)
                return data_ver

        await DataBackend.start_refresh_table(self._data_backend_ctx,
                                              execution_spec, table_def,
                                              data_ver, alias=table_alias)

        return data_ver

    async def _get_table_def_and_connector_ref(
            self,
            id: TableIDv2) -> Tuple[TableDefinition, ConnectorExecutionRef]:
        # Get the v2 table definition from the metadata db
        table_def = await self.get_table_v2(id)
        if not table_def:
            raise TableDoesNotExistError(id)

        connector_id = table_def.get_connector_id()
        connector_spec = await self._execution_connector(connector_id)

        return table_def, connector_spec

    async def get_col_stats_and_sample_rows(self,
                                            id: TableIDv2) -> ColStatsResponse:
        """Get col stats and sample rows for a table. This will return exact
        stats if present, otherwise approx stats. It will always return sample
        rows.
        """
        table_def, connector_ref = (await
                                    self._get_table_def_and_connector_ref(id))
        return await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            get_table_col_stats_and_sample_rows_activity,
            [connector_ref, table_def],
            f'table-stats:{self._project_id}',
        )

    async def get_col_stats_at_version(
            self, id: TableIDv2,
            version: TableDataVersion) -> ColStatsAtVersion:
        """
        This method gets cached column stats at a specific version, used for
        getting stats at training time to compare to stats at prediction time.

        It returns ColStatsAtVersion which contains
            - stats_dict: Dict[ColumnName, ColumnStats]
            - version: TableDataVersion
            - approx: bool
        """

        table_def, _connector_spec = (await
                                      self._get_table_def_and_connector_ref(id))
        table_fingerprint = table_def.fingerprint()
        # DG-native: read exact min col stats from BackendCacheManager
        # directly (legacy Backend.load_exact_min_col_stats_if_present
        # was a thin load_min_col_stats_cache + get_min_col_stats
        # wrapper around the same cache).
        cache = get_active_cache()
        cache.load_min_col_stats_cache(
            connector_name=table_fingerprint.connector_name,
            table_fingerprint=table_fingerprint,
            data_version=version,
        )
        exact_col_stats = cache.get_min_col_stats(
            table_fingerprint=table_fingerprint,
            stats_dict=ColStatsSet.UI.get_stats_types(table_def),
            data_version=version,
            approx=False,
        )

        return _parse_col_stats(version, exact_col_stats, table_def)

    @measure_blocking_code
    async def infer_table_metadata(
        self,
        request: table_api.TableMetadataRequest,
    ) -> table_api.TableMetadataResponse:

        source_table, _, connector_id = extract_source_table_type(
            request.source_table)
        assert connector_id is not None
        reference = await self._execution_connector(connector_id)
        # A Custom SQL table inherits stypes from the sibling Kumo tables its
        # query reads; resolve those hints here (state DB is reachable) and
        # append them as a 3rd activity arg. Physical tables keep the prior
        # 2-arg ``[reference, request]`` arity so a rolling deploy to an old
        # data-backend worker (2-param signature) still infers physical schemas.
        activity_inputs: List[Any] = [reference, request]
        if isinstance(source_table, CustomSQLSourceTable):
            # Reject a non-warehouse connector before dispatching inference --
            # the same authoritative gate the create/validate paths apply,
            # resolved from credential-free connector metadata. Reuse the
            # resolved dialect for the stype hints so no second connector read is
            # needed.
            connector = await self._require_state_services(
            ).connectors.get_for_project(self._project_id, connector_id,
                                         _current_user_ctx.get())
            connector_type = ConnectorType(connector.config.type.value)
            authoritative_sql_connector_type(connector_type)
            activity_inputs.append(await _sql_reference_stype_hints(
                self._require_state_services(), _current_user_ctx.get(),
                self._project_id, connector_id, source_table.sql,
                connector_type=connector_type))
        return await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            infer_table_metadata_activity,
            activity_inputs,
            f'infer-table-metadata:{self._project_id}',
        )

    @measure_blocking_code
    async def validate_table(
            self, definition: TableDefinition,
            max_table_size_limits: Dict[str, int] = {}) -> None:
        reference = await self._execution_connector(
            definition.get_connector_id())
        if definition.is_sql:
            # A Custom SQL Table has no physical source table: validate its query
            # schema in the dedicated activity (connector resolved via the
            # broker) and fail creation loudly on an invalid definition. The
            # size/permission checks are skipped: read access is exercised by
            # describing the query under least-privilege credentials, and result
            # size is bounded at materialization.
            reference_stypes = await _sql_reference_stype_hints(
                self._require_state_services(),
                _current_user_ctx.get(), self._project_id,
                definition.get_connector_id(), definition.source_table.sql)
            response = await DataBackend._execute_single_activity(
                self._data_backend_ctx,
                validate_custom_sql,
                [reference,
                 _sql_resolve_request(definition), reference_stypes],
                f'validate-custom-sql:{self._project_id}',
                execution_timeout=timedelta(minutes=5),
            )
            if response.table_definition is None:
                raise ValidationException(
                    'Custom SQL Table Validation Failure',
                    ' '.join(error.message for error in response.errors)
                    or 'The SQL query is not a valid Custom SQL Table.')
            return
        await DataBackend._execute_single_activity(
            self._data_backend_ctx,
            validate_table_definition_activity,
            [reference, definition, max_table_size_limits],
            f'validate-table:{self._project_id}',
        )

    async def get_many_table_configs_for_sdk(
        self,
        table_name_to_dbid: Dict[TableName, FPID],
    ) -> Tuple[List[ConnectorSpec], List[TableReferenceMetadata]]:
        """
        Creates the connector and table configs for each table
        and unqiue connector in a graph.
        If there are tables with mixed connectors, creates the
        connector config for each unique connector.

        Args:
            table_name_to_dbid Dict[TableName, FPID]: Metadata
            used to fetch all the tables in a graph.
        Response:
            Tuple[List[ConnectorSpec], List[TableReferenceMetadata]]
        """
        # Create a mapping of fpid -> table_alias to fetch names of tables in a graph.
        # This is specifically for the case when a graph is saved in the UI from the SDK
        # but the individual tables are not. In this case, the tables don't have an alias
        # in our DB, but have an alias in the graph.
        table_fpids: Set[FPID] = set()
        table_fpid_to_name_in_graph: Dict[FPID, str] = {}
        for name, fpid in table_name_to_dbid.items():
            table_fpid_to_name_in_graph[fpid] = name
            table_fpids.add(fpid)

        db_entries = await self._require_table_service().get_table_entries(
            self._project_id, table_fpids)
        table_confs: List[TableReferenceMetadata] = []
        connector_confs_set = set()
        connector_spec_arr: List[ConnectorSpec] = []
        for fpid, table_entry in db_entries.items():
            table_def = table_entry.config

            connector_conf, src_tbl_conf = self.get_source_table_and_connector_metadata(
                table_entry,
                TableIDv2("NA.NA-" + fpid),
                table_fpid_to_name_in_graph,
            )

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
                    **table_metadata
                },
                table_alias_in_graph=table_fpid_to_name_in_graph[fpid],
            )
            # NOTE: With unnamed tables, the same table definition can be
            # re-used under different aliases. Graphs and pqueries refer
            # to tables by an alias. So having both the FPID and alias
            # clarifies which table we are exactly interested in.
            if table_entry.alias:
                table_config.table_alias = table_entry.alias

            table_confs.append(table_config)
            connector_confs_set.add(connector_conf)

        # Since connector duplicates have been removed,
        # convert the connector configs to dictionaries.
        for config in connector_confs_set:
            config_as_dict = dict(config)
            if "connector_id" in config_as_dict:
                connector_id = ConnectorID(config_as_dict["connector_id"])
            elif "name" in config_as_dict:
                connector_id = ConnectorID(config_as_dict["name"])
            else:
                raise ValueError(
                    f"Connector with config {config_as_dict} does " +
                    "not have a valied connector ID.")
            connector = await self._require_state_services(
            ).connectors.source_table_connector_for_project(
                self._project_id,
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

        return (connector_spec_arr, table_confs)

    def get_source_table_and_connector_metadata(
        self,
        table_db_entry: TableDBEntryV2,
        table_id: TableIDv2,
        table_fpid_to_name_in_graph: Dict[FPID, str],
    ) -> Tuple[Tuple[Tuple[str, Any], ...], Dict[str, Any]]:
        """
        Generate a table's and its connector's config. The generated
        configs vary slightly based on the connector type.
        Every source table references a durable project-owned connector.
        """
        source_table = table_db_entry.config.source_table
        # Initialize connector config as a set of tuples to be able to avoid duplicate
        # connector configs.
        connector_config: Tuple[Tuple[str, Any], ...] = ()
        source_table_config = {}
        source_table_type = source_table.data_source_type
        if source_table_type == DataSourceType.S3:
            connector_id = source_table.connector_id
            source_table_name = source_table.table_name
            file_type = source_table.file_type
            connector_config = (("data_source_type", DataSourceType.S3.value),
                                ("connector_id", connector_id))
            source_table_config["connector_id"] = connector_id
            source_table_config["table_name"] = source_table_name
            source_table_config["file_type"] = file_type.value

        elif source_table_type == DataSourceType.SNOWFLAKE:
            connector_id = source_table.snowflake_connector_id
            connector_config = (
                ("name", connector_id),
                ("data_source_type", DataSourceType.SNOWFLAKE.value),
            )
            source_table_config["table_name"] = source_table.table
            source_table_config["connector_id"] = connector_id
        elif source_table_type == DataSourceType.DATABRICKS:
            connector_id = source_table.databricks_connector_id
            connector_config = (
                ("name", connector_id),
                ("data_source_type", DataSourceType.DATABRICKS.value),
            )
            source_table_config["table_name"] = source_table.table
            source_table_config["connector_id"] = connector_id
        elif source_table_type == DataSourceType.BIGQUERY:
            connector_id = source_table.bigquery_connector_id
            connector_config = (
                ("name", connector_id),
                ("data_source_type", DataSourceType.BIGQUERY.value),
            )
            source_table_config["table_name"] = source_table.table_name
            source_table_config["connector_id"] = connector_id
        elif source_table_type == DataSourceType.GCS:
            connector_id = source_table.connector_id
            source_table_name = source_table.source_table_name
            file_type = source_table.file_type
            connector_config = (("data_source_type", DataSourceType.GCS.value),
                                ("connector_id", connector_id))
            source_table_config["connector_id"] = connector_id
            source_table_config["table_name"] = source_table_name
            source_table_config["file_type"] = file_type.value
        elif source_table_type == DataSourceType.ADLS:
            connector_id = source_table.connector_id
            source_table_name = source_table.source_table_name
            file_type = source_table.file_type
            connector_config = (("data_source_type", DataSourceType.ADLS.value),
                                ("connector_id", connector_id))
            source_table_config["connector_id"] = connector_id
            source_table_config["table_name"] = source_table_name
            source_table_config["file_type"] = file_type.value
        elif source_table_type == DataSourceType.CUSTOM_SQL:
            # A Custom SQL table has no physical source table: carry the query,
            # the synthetic table name, and the backing warehouse connector id
            # so the SDK-script generator emits `Table.from_sql(...)`. The
            # connector's concrete warehouse type is not recorded on the source
            # table, so it stays CUSTOM_SQL here.
            connector_id = source_table.connector_id
            connector_config = (
                ("name", connector_id),
                ("data_source_type", DataSourceType.CUSTOM_SQL.value),
            )
            source_table_config["table_name"] = source_table.table
            source_table_config["connector_id"] = connector_id
            source_table_config["sql"] = source_table.sql
        else:
            raise ValueError(
                f"Invalid {source_table_type} for table with ID {table_id}.")

        source_table_config["table_name_alias"] = table_db_entry.alias

        if not table_db_entry.alias:
            # In case a table is unnamed, use the
            # table alias the graph uses.
            source_table_config[
                "table_name_alias"] = table_fpid_to_name_in_graph[
                    table_db_entry.id]
        return (connector_config, source_table_config)


@blocking_func()
def _get_source_table_metadata(source_table_name: str,
                               connector: Connector) -> TableMetadataConfig:
    original_meta_tables: List[TableMetadataConfig] = \
        connector.get_table_metadata(
            table_names=[source_table_name],
            skip_inferral=True,
            skip_cache=True,
            ignore_errors=True)
    if len(original_meta_tables) == 0:
        raise PublicException(message=(
            f"Failed to validate table from '{source_table_name}' "
            f"in connector '{connector.name}'. "
            "Please check whether table was accidentally dropped, "
            "renamed or emptied since it was first imported into Kumo."))
    return original_meta_tables[0]


def _sql_validation_error_message(error: Exception) -> str:
    """Return an allowlisted Custom SQL inference failure message."""
    if isinstance(error, ConnectorError):
        return sql_query_error_message(error)
    if isinstance(error, PublicException):
        return error.message
    return str(error)


def _sql_resolve_error(error: Exception) -> ApiValidationError:
    r"""Map known failures to stable public errors and redact unknown details."""
    if isinstance(error, ConnectorError):
        title = 'Warehouse query failed'
        message = sql_query_error_message(error)
    elif isinstance(error,
                    (UnsupportedColumnTypeError, UnsupportedInDiskgraphError)):
        title = 'Unsupported column type'
        message = (error.message if isinstance(
            error, UnsupportedInDiskgraphError) else str(error))
    elif isinstance(error, (DuplicateColumnsError, CustomSqlOutputSchemaError)):
        title = 'Invalid output schema'
        message = str(error)
    elif isinstance(error, PublicException):
        title = 'Invalid Custom SQL table'
        message = error.message
    elif type(error) is ValueError:
        title = 'Invalid Custom SQL table'
        message = str(error)
    else:
        log.error('Unexpected SQL resolution failure (error_type=%s)',
                  type(error).__name__)
        title = 'Custom SQL resolution failed'
        message = 'Custom SQL resolution failed unexpectedly.'
    return ApiValidationError(title=title, message=message)


def _sql_preview_rows(
    sample: Optional[pd.DataFrame],
    columns: List[Column],
    num_rows: int,
) -> Optional[List[List[Optional[str]]]]:
    r"""Return an ordered bounded preview, or ``None`` when cost-gated."""
    if sample is None:
        return None
    col_names = [col.name for col in columns]
    rows_df = sample.head(num_rows).reindex(columns=col_names)
    return stringify_sample_dataframe(rows_df).values.tolist()


def authoritative_sql_connector_type(
        resolved_connector_type: ConnectorType) -> ConnectorType:
    r"""Accept a server-resolved SQL warehouse type, never a client assertion."""
    sqlglot_dialect(resolved_connector_type)
    return resolved_connector_type


def canonicalize_custom_sql_source(
        table_def: TableDefinition,
        connector_type: ConnectorType) -> TableDefinition:
    r"""Stamp canonical identity, lineage, and server-owned warehouse type.

    Normalized SQL anchors path and cache identity. Connector type and
    referenced tables are always derived server-side before persistence.
    """
    if not table_def.is_sql:
        return table_def
    src = table_def.source_table
    assert isinstance(src, CustomSQLSourceTable)
    resolved_type = authoritative_sql_connector_type(connector_type)
    canonical = TableName(custom_sql_synthetic_name(src.sql, connector_type))
    refs = referenced_tables_or_none(src.sql, connector_type) or []
    if (canonical == src.table and refs == list(src.referenced_tables)
            and src.connector_type == resolved_type):
        return table_def
    return dataclasses.replace(
        table_def,
        source_table=dataclasses.replace(src, table=canonical,
                                         referenced_tables=refs,
                                         connector_type=resolved_type))


def _sql_resolve_request(
        definition: TableDefinition) -> table_api.SqlTableResolveRequest:
    r"""Adapt a persisted SQL definition to the discovery request."""
    assert isinstance(definition.source_table, CustomSQLSourceTable)
    return table_api.SqlTableResolveRequest(
        source_table=definition.source_table,
        column_hints=([
            table_api.ColumnMetadataRequest(
                col.name, col.stype, col.dtype, col.timestamp_format,
                col.multicategorical_delimiter) for col in definition.cols
        ] or None),
        pkey=definition.pkey,
        time_col=definition.time_col,
        end_time_col=definition.end_time_col,
    )


async def _validate_and_preview_sql_table(
    request: table_api.SqlTableResolveRequest,
    connector: Connector,
    connector_type: ConnectorType,
    reference_stypes: Optional[Dict[str, Dict[str, Stype]]] = None,
) -> table_api.SqlTableResolveResponse:
    r"""Resolve Custom SQL through one schema and at most one sample pass.

    Raw SQL is retained for persistence and identity. After write and namespace
    gates, only a terminator-safe copy is executed; the synthetic source alias
    is never looked up as a physical table.
    """
    src = request.source_table
    resolved_type = authoritative_sql_connector_type(connector_type)
    raw_sql = src.sql
    advisory_read_only_check(raw_sql, connector_type)
    canonical = TableName(custom_sql_synthetic_name(raw_sql, connector_type))
    refs = validate_connector_sql_scope(raw_sql, connector_type, connector)
    sql = subquery_safe_sql(raw_sql, connector_type)
    resolved_source = dataclasses.replace(src, table=canonical,
                                          referenced_tables=refs,
                                          connector_type=resolved_type)
    execution_source = dataclasses.replace(resolved_source, sql=sql)
    metadata_request = table_api.TableMetadataRequest(
        cols=request.column_hints or [],
        source_table=execution_source,
        pkey=request.pkey,
        time_col=request.time_col,
        end_time_col=request.end_time_col,
    )
    metadata, source_metadata, table = (await
                                        _resolve_table_metadata_with_connector(
                                            metadata_request, connector,
                                            connector_type, reference_stypes,
                                            request.column_hints is None))
    assert source_metadata is not None
    table_def = TableDefinition(
        cols=metadata.cols,
        source_table=resolved_source,
        pkey=metadata.pkey,
        time_col=metadata.time_col,
        end_time_col=metadata.end_time_col,
    )
    execution_def = dataclasses.replace(table_def,
                                        source_table=execution_source)
    _validate_col_names_v2(execution_def)
    _validate_cols_are_unique_v2(execution_def)
    _validate_pkey_and_time_col_v2(execution_def)
    sql_sample = table._read_inference_sample()
    await _validate_table_metadata_v2.coroutine(execution_def, connector,
                                                source_metadata)

    stats_version = custom_sql_stats_version(
        sql, connector_type, {c.name: c.dtype
                              for c in table.data_cols})
    await table.persist_sql_preview_col_stats.coroutine(
        stats_version, ColStatsSet.UI.get_stats_types(execution_def))

    preview_rows = _sql_preview_rows(sql_sample, metadata.cols,
                                     request.preview_rows)
    preview_note: Optional[str] = None
    if preview_rows is None:
        preview_rows = []
        preview_note = 'Preview skipped — query too large.'
    return table_api.SqlTableResolveResponse(
        table_definition=table_def,
        preview_rows=preview_rows,
        preview_note=preview_note,
    )


async def _sql_reference_stypes(
    table_service: TableService,
    project_id: ProjectID,
    connector_id: ConnectorID,
    sql: str,
    connector_type: ConnectorType,
) -> Dict[str, Dict[str, Stype]]:
    r"""Column stypes of referenced external tables that are independently Kumo
    Tables, for the Custom SQL stype-inheritance pre-pass.

    Resolves each external table the query reads to a raw (non-SQL) Kumo Table
    on the same connector, keying its user/inferred column stypes by
    dialect-normalized table and column name so ``source_table_for_col`` results
    reconcile against them. Best-effort: unparseable SQL, no referenced table
    being a Kumo Table, an ambiguous bare name (shared by more than one raw
    table), or any lookup failure yields an empty mapping (inference then falls
    back to heuristics). Never raises.
    """
    referenced = referenced_tables_or_none(sql, connector_type)
    if not referenced:
        return {}
    # Match on the bare (leaf) table name: a Kumo table stores its source name
    # unqualified, while the SQL may reference it qualified, so reconcile on the
    # final component. Drop a leaf shared by two distinct *referenced* tables
    # (e.g. ``db1.orders`` + ``db2.orders``): it can't be resolved to one raw
    # table, so leave those columns to the heuristic.
    ref_leaf_counts: Dict[str, int] = {}
    for name in referenced:
        leaf = normalized_sql_identifier_leaf(name, connector_type)
        ref_leaf_counts[leaf] = ref_leaf_counts.get(leaf, 0) + 1
    wanted = {leaf for leaf, count in ref_leaf_counts.items() if count == 1}
    if not wanted:
        return {}
    try:
        entries = await table_service.list_tables(project_id, connector_id)
        # Inherit only from raw Kumo Tables (never another SQL table) whose bare
        # name matches a reference. Count matches per leaf first so a name shared
        # by more than one raw table is dropped as ambiguous — even if one of the
        # duplicates has no inheritable stypes.
        raw: List[Tuple[str, object]] = []
        for entry in entries:
            config = entry.config
            if config.is_sql:
                continue
            leaf = normalized_sql_identifier_leaf(
                config.get_source_table_name(), connector_type)
            if leaf in wanted:
                raw.append((leaf, config))
        leaf_counts: Dict[str, int] = {}
        for leaf, _ in raw:
            leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
        result: Dict[str, Dict[str, Stype]] = {}
        for leaf, config in raw:
            if leaf_counts[leaf] != 1:  # ambiguous: refuse to guess
                continue
            col_stypes = {
                normalized_sql_identifier(col.name, connector_type):
                _coerce_stype(col.stype)
                for col in config.cols
                # An ``unsupported`` stype is not inheritable (and would raise in
                # the dtype-compatibility check), so drop it.
                if col.stype is not None and col.stype != Stype.unsupported
            }
            if col_stypes:
                result[leaf] = col_stypes
        return result
    except Exception:
        # Inheritance is a best-effort enrichment: a state-DB read or coercion
        # failure must degrade to heuristic inference, never block the request.
        log.warning(
            "Custom SQL stype inheritance: failed to resolve "
            "referenced Kumo tables; using heuristics.", exc_info=True)
        return {}


async def _sql_reference_stype_hints(
    state_services: StateServices,
    user: Any,
    project_id: Union[ProjectID, str],
    connector_id: ConnectorID,
    sql: str,
    connector_type: Optional[ConnectorType] = None,
) -> Optional[SqlReferenceStypes]:
    r"""Resolve Custom SQL's dialect and optional stype-inheritance hints.

    Computed REST/driver-side because it reads sibling tables from the state DB
    (``_sql_reference_stypes``), which a bare connector activity cannot. SQL
    tables only -- physical tables never call this. Returns ``None`` only when
    the dialect cannot be resolved; an empty ``stypes`` map still transports
    the authoritative connector type to the activity.
    """
    try:
        project_id = ProjectID(project_id)
        if connector_type is None:
            connector = await state_services.connectors.get_for_project(
                project_id, connector_id, user)
            connector_type = ConnectorType(connector.config.type.value)
        stypes = await _sql_reference_stypes(state_services.tables, project_id,
                                             connector_id, sql, connector_type)
    except Exception:
        # Resolving inheritance hints is a best-effort enrichment: a dialect or
        # state-DB read failure must degrade to heuristic inference, never block
        # metadata inference / validation.
        if connector_type is None:
            return None
        stypes = {}
    return SqlReferenceStypes(connector_type=connector_type, stypes=stypes)


def _build_sql_column_summaries(
    column_metadata: ColumnMetadataDict,
    stats: ColStatsAtVersion,
    sample_rows: Optional[Dict[str, List[Any]]] = None,
) -> List[ColumnSummary]:
    """Column summaries for a Custom SQL table's stats/preview response.

    Unlike the physical ``_build_column_summaries``, a column without stats
    (e.g. a cost-gated preview that skipped the sample) yields ``None``
    statistics rather than raising, so a schema-only preview still renders.
    """
    columns: List[ColumnSummary] = []
    for data in column_metadata.values():
        name = data['name']
        assert isinstance(name, str)
        stype = data['stype']
        dtype = data['dtype']
        assert isinstance(stype, Stype)
        assert isinstance(dtype, Dtype)
        columns.append(
            ColumnSummary(
                name=name,
                stype=stype.title_case(),
                dtype=dtype.title_case(),
                sample_rows=([] if sample_rows is None else sample_rows.get(
                    name, [])),
                statistics=stats.stats_dict.get(name),
            ))
    return columns


@blocking_func(DATA_BACKEND_THREAD_POOL)
def _read_sql_snapshot_sample_rows(
    table_def: TableDefinition,
    data_version: str,
    num_rows: int,
) -> Optional[Dict[str, List[Optional[str]]]]:
    """Read sample rows for a materialized Custom SQL table from its ingested
    snapshot parquet from the legacy DiskGraph local-ingest output, never the
    warehouse.

    Returns a ``{col_name: [values]}`` dict (stringified, nulls as ``None``), or
    ``None`` when the snapshot parquet is not present so the caller renders
    stats without sample rows.
    """
    ingest_dir = get_diskgraph_local_ingest_path(table_def.fingerprint(),
                                                 str(data_version))
    col_names = [col.name for col in table_def.cols]
    if not fsutil.isdir(ingest_dir):
        return None
    parts = sorted(entry for entry in fsutil.listdir(ingest_dir)
                   if entry.endswith('.parquet'))
    if not parts:
        return None

    # Bounded read: pull at most ``num_rows`` rows and only the needed columns,
    # so a large materialization shard never loads a whole parquet file into the
    # REST process (OOM/slow-request guard).
    frames: List[pd.DataFrame] = []
    remaining = num_rows
    for part in parts:
        if remaining <= 0:
            break
        parquet_file = pq.ParquetFile(osp.join(ingest_dir, part))
        present = [c for c in col_names if c in parquet_file.schema_arrow.names]
        for batch in parquet_file.iter_batches(batch_size=remaining,
                                               columns=present or None):
            frames.append(batch.to_pandas())
            remaining -= batch.num_rows
            break  # first bounded batch already yields <= remaining rows
    sample = pd.concat(frames) if frames else pd.DataFrame(columns=col_names)
    sample = sample.reindex(columns=col_names)
    sample = sample.astype('str').mask(sample.isnull(), None)
    return {name: sample[name].tolist() for name in col_names}


async def _get_sql_col_stats_and_sample_rows(
    table_def: TableDefinition,
    connector_spec: ConnectorOpenSpec,
    col_stats_set: ColStatsSet,
) -> ColStatsResponse:
    """Col stats + sample rows for a Custom SQL table.

    Post-materialization serves exact stats and sample rows from the ingested
    snapshot; pre-materialization serves the persisted Validate-time approx
    stats keyed by definition identity, with no sample rows (button-only). The
    warehouse is never read through the synthetic alias.
    """
    cache = get_active_cache()
    connector_id = table_def.get_connector_id()
    source_name = str(table_def.get_source_table_name())
    column_metadata = _get_column_metadata(table_def)
    request = col_stats_set.get_stats_types(table_def)

    # ``table_version_cache`` is keyed by the SQL-derived synthetic name only,
    # so a non-None hit means SOME same-SQL definition is materialized — not
    # necessarily this cols variant. When something is materialized, resolve
    # THIS definition's content ``data_version`` (which folds in the output
    # cols) and read the snapshot at exactly that version; a cols-only sibling
    # yields no stats at this version and falls through to the preview. The
    # cheap synthetic-name check gates the per-open metadata probe so the common
    # unmaterialized editing loop never pays it.
    ingested_version = await cache.table_version_cache.get.coroutine(
        connector_id, source_name)

    if ingested_version is not None:
        data_version = await get_table_data_version(connector_spec, source_name,
                                                    table_def=table_def)
        exact = await DataBackend.get_min_col_stats(
            connector_spec, table_def, request, NUM_ROWS_FOR_APPROX_STATS,
            check_refresh_status=False, compute_missing_stats=False,
            table_data_version=TableDataVersion(data_version))
        if any(exact.values()):
            stats = _parse_col_stats(version=None, col_stats=exact,
                                     table_def=table_def)
            sample_rows = await _read_sql_snapshot_sample_rows.coroutine(
                table_def, data_version, DEFAULT_NUM_SAMPLE_ROWS)
            columns = _build_sql_column_summaries(column_metadata, stats,
                                                  sample_rows)
            return ColStatsResponse(statistics_available=True, columns=columns,
                                    approx=stats.approx,
                                    rows_used=stats.num_rows_used)

    # Pre-materialization: persisted Validate-time approx stats, keyed by
    # definition identity (not the churny content data_version).
    stats_version = custom_sql_stats_version(
        table_def.source_table.sql, _connector_type_from_spec(connector_spec),
        {c.name: c.dtype
         for c in table_def.cols})
    fingerprint = table_def.fingerprint()
    cache.load_col_stats_cache(fingerprint, stats_version)
    approx = cache.get_col_stats(fingerprint, request, stats_version,
                                 approx=True)
    stats = _parse_col_stats(version=None, col_stats=approx,
                             table_def=table_def)
    columns = _build_sql_column_summaries(column_metadata, stats)
    return ColStatsResponse(statistics_available=bool(approx), columns=columns,
                            approx=True, rows_used=stats.num_rows_used)


# ---------------------------------------------------------------------------
# V2-native validators.
#
# These mirror the legacy v1 validators but operate on `TableDefinition`
# directly — no v1 downgrade. The v2 entry point (`TableLogic.validate_table`)
# calls these instead of downgrading to the v1 family. Message format
# matches the legacy v1 behavior.
#
# User-input errors raise the typed `ValidationException` (allowlisted in
# `_SINGLE_ACTIVITY_PUBLIC_ERROR_TYPES`) rather than v1's bare `ValueError`:
# these validators run inside data-backend single activities, and
# `_raise_safe_single_activity_failure` masks any non-allowlisted exception
# type as a generic InternalError. The typed exception is what lets the
# hand-written, actionable message reach the caller with a 422 instead of
# a generic 500 (kumo-ai/kumo#29794).
#
# Differences from v1:
#   * `_validate_pkey_and_time_col_v2` drops the `if not args.is_view:`
#     guard — v2 doesn't model views.
#   * Error messages that referenced `args.table_name` use the table's
#     source name (`get_source_table_name()`); v2 has no per-graph alias.
# ---------------------------------------------------------------------------


def _v2_table_label(definition: TableDefinition) -> str:
    """Stand-in for v1's `args.table_name` in error messages — v2 has no
    per-graph alias on the definition itself, so use the source name."""
    return str(definition.get_source_table_name())


def _validate_col_names_v2(definition: TableDefinition) -> None:
    # Raises a bare ValueError (from the shared `string.validate_sql_id`)
    # if col.name does not fit psql char requirements, so this one path is
    # still masked by the single-activity wrapper; wrapping the shared
    # utility is a follow-up on kumo-ai/kumo#29794.
    for col in definition.cols:
        string.validate_sql_id(col.name, True)

    reserved_pkey_names = [
        "ENTITY", "TARGET", "ASSUMPTION", "ENTITY_PRECOMPUTE"
    ]
    for col in definition.cols:
        if col.name.upper() in reserved_pkey_names:
            if col.name == definition.pkey:
                raise TableInvalidColumnNameError(
                    message=f"Column '{col.name}' conflicts with Kumo's "
                    "reserved internal column names. "
                    f"A primary key column cannot be named: {reserved_pkey_names}."
                )


def _validate_cols_are_unique_v2(definition: TableDefinition) -> None:
    seen_col_names: Set[str] = set()
    label = _v2_table_label(definition)
    for col in definition.cols:
        if col.name in seen_col_names:
            raise ValidationException(
                title=DUPLICATE_COLUMN_NAMES_TITLE,
                message="There should not be any duplicate col names, "
                "but received two cols with name "
                f"'{label}.{col.name}'.")
        seen_col_names.add(col.name)


def _validate_pkey_and_time_col_v2(definition: TableDefinition) -> None:
    # v2 doesn't model views — `if not args.is_view:` from v1 is unreachable
    # here, so we run pkey/time-col validation unconditionally.
    col_dict = {col.name: col for col in definition.cols}
    label = _v2_table_label(definition)

    if definition.pkey is not None and len(definition.pkey) != 0:
        if definition.pkey not in col_dict:
            raise ValidationException(title=INVALID_PRIMARY_KEY_TITLE,
                                      message=PKEY_PRESENT_IN_COLS)
        if _coerce_stype(col_dict[definition.pkey].stype) != Stype.ID:
            raise ValidationException(title=INVALID_PRIMARY_KEY_TITLE,
                                      message=PKEY_IS_ID)

    for time_col in [definition.time_col, definition.end_time_col]:
        if time_col is not None and len(time_col) != 0:
            if time_col not in col_dict:
                raise ValidationException(title=INVALID_TIME_COLUMN_TITLE,
                                          message=TIME_COL_PRESENT_IN_COLS)

            if time_col == definition.pkey:
                raise ValidationException(title=INVALID_TIME_COLUMN_TITLE,
                                          message=TIME_COL_NEQ_PKEY)

            if _coerce_stype(col_dict[time_col].stype) != Stype.timestamp:
                raise ValidationException(
                    title=INVALID_TIME_COLUMN_TITLE,
                    message=f"Time Column '{label}.{time_col}' must "
                    f"have semantic data type of '{Stype.timestamp}', but "
                    f"currently has '{col_dict[time_col].stype}'")


@blocking_func()
def _validate_table_metadata_v2(
    definition: TableDefinition,
    connector: Connector,
    source_table_metadata: TableMetadataConfig,
) -> None:
    log.info(f"Validating table metadata for table {definition}")
    label = _v2_table_label(definition)
    try:
        table = table_args_to_kumo_table(definition, connector)
    except ValueError as error:
        raise ValidationException(
            title='Invalid table definition',
            message=str(error),
        ) from None
    log.info(f"Table {definition} has Kumo backend representation {table}")

    # SQL callers pass the ordered metadata already used for inference, so
    # compatibility validation never describes the query a second time.
    input_cols = source_table_metadata.all_cols()
    input_dtype_dict = {col.name: col.dtype for col in input_cols}
    unsupported_cols = [
        col.name for col in definition.cols
        if input_dtype_dict.get(col.name) == Dtype.unsupported and (
            col.dtype is None or _coerce_dtype(col.dtype) == Dtype.unsupported)
    ]
    if unsupported_cols:
        formatted_cols = ', '.join(f"'{name}'" for name in unsupported_cols)
        noun = 'Column' if len(unsupported_cols) == 1 else 'Columns'
        raise ValidationException(
            title='Unsupported column type',
            message=f"{noun} {formatted_cols} in table {label} "
            "have unsupported source data types. Exclude them from the Kumo "
            "table definition, declare a data type for them, or cast them "
            "to supported output types in a Custom SQL query.")

    for col in definition.cols:
        if col.dtype is None:
            raise ValidationException(
                title=MISSING_COLUMN_DATA_TYPE_TITLE,
                message=f"Column {col} has non-specified data type.")
        if col.name not in input_dtype_dict:
            raise ValidationException(
                title=TABLE_COLUMN_NOT_FOUND_TITLE,
                message=f"Column {col} does not exist in table "
                f"{label} behind connector {connector}. "
                f"Please check your source data and try again.")
        table_name = (source_table_metadata.table_name
                      if table.table_name == "NA" else table.table_name)
        col_dtype = _coerce_dtype(col.dtype)
        if (input_dtype_dict[col.name] != Dtype.unsupported
                and not can_cast_safely(col_dtype, input_dtype_dict[col.name])):
            if (col_dtype == Dtype.string.value
                    and input_dtype_dict[col.name] == Dtype.stringlist):
                continue
            raise ValidationException(
                title=INVALID_COLUMN_DATA_TYPE_TITLE,
                message=f"Data type of column '{col.name}' in table "
                f"'{table_name}' was selected to be "
                f"{col_dtype.title_case()} but should be "
                f"{input_dtype_dict[col.name].title_case()} based on "
                f"sample data. Please update the data type of the column "
                f"in the table.")

    # Perform regular table validations without letting their bare ValueError
    # cross the single-activity boundary and become a generic
    # InternalError.
    try:
        table.validate(require_key=False, ignore_encoders=True)
    except ValueError as error:
        raise ValidationException(
            title='Invalid table definition',
            message=str(error),
        ) from None


@blocking_func()
def _validate_table_size_v2(
    definition: TableDefinition,
    connector: Connector,
    max_table_size_limits: Dict[str, int],
) -> None:
    log.info(f"Validating table size for table {definition}")
    source_table_name = str(definition.get_source_table_name())
    base_col_configs = [
        definition.col_to_base_config(col.name) for col in definition.cols
    ]
    # `col_to_base_config` returns Optional; for cols we just iterated over,
    # the value is always present.
    base_col_configs = [c for c in base_col_configs if c is not None]
    connector_type = connector.__class__.__name__.removesuffix("Connector")

    if (max_size := max_table_size_limits.get(connector_type.lower())):
        try:
            table_size_bytes = connector._get_selected_table_source_size(
                source_table_name, base_col_configs)
        except NotImplementedError:
            table_size_bytes = 0  # ignore if not set
        log.info(f"Size of the table is {table_size_bytes}, max is {max_size}")
        if table_size_bytes > max_size:
            raise ValidationException(
                TABLE_SIZE_EXCEEDS_LIMIT_TITLE,
                TABLE_SIZE_EXCEEDS_LIMIT_MESSAGE.format(
                    source_table_name, connector_type, connector.name,
                    format_bytes(table_size_bytes)))


@blocking_func()
def _validate_table_permissions_v2(
    definition: TableDefinition,
    connector: Connector,
) -> None:
    log.info(f"Validating read permissions for table {definition}")
    if isinstance(connector, SQLConnector):
        source_table_name = str(definition.get_source_table_name())
        try:
            connector.check_table_read_permissions(source_table_name)
        except NotImplementedError:
            return


def _table_read_error_v2(table: TableDefinition, err: Exception) -> Exception:
    """v2-native sibling of the (removed) v1 ``_table_read_error``."""
    return ValidationHTTPException(
        TABLE_READ_ERROR_MESSAGE.format(str(table.get_source_table_name()),
                                        table.get_connector_id(),
                                        TABLE_READ_PROVIDER_ERROR), err)


def _get_column_metadata(table_def: TableDefinition) -> ColumnMetadataDict:
    # Coerce stype/dtype to the extended types so downstream
    # ``isinstance(..., Stype)`` checks pass.
    return {
        col.name: {
            'name': col.name,
            'dtype': _coerce_dtype(col.dtype),
            'stype': _coerce_stype(col.stype),
        }
        for col in table_def.cols
    }


async def _get_sample_rows(table_def: TableDefinition,
                           connector_spec: ConnectorSpec):
    try:
        column_sample_rows: Dict[str, List[Optional[str]]] = {}

        # Get sample row data
        col_names = [col.name for col in table_def.cols]
        base_col_configs = [
            cfg for cfg in (table_def.col_to_base_config(col.name)
                            for col in table_def.cols) if cfg is not None
        ]
        connector = await get_connector_from_spec_async(
            connector_spec, resolved_factory=get_connector_from_spec)
        async with async_connector_lifecycle(connector):
            table_sample_rows_df = await get_sample_rows_raw.coroutine(
                connector=connector,
                source_table_name=str(table_def.get_source_table_name()),
                cols=base_col_configs,
                num_rows=DEFAULT_NUM_SAMPLE_ROWS,
            )
        table_sample_rows = table_sample_rows_df.values.tolist()

        # Add columns
        for i, col_name in enumerate(col_names):
            sample_rows = [row[i] for row in table_sample_rows]
            column_sample_rows[col_name] = sample_rows

        return column_sample_rows
    except ValueError as e:
        log.warning('Unable to sample source table values (error_type=%s)',
                    type(e).__name__)
        raise ValueError(
            'Sample rows contain invalid values. This may not block table '
            'ingestion, but filtering invalid values in the data plane is '
            'recommended.') from None
    except (FileNotFoundError, PermissionError, ConnectorError) as e:
        raise _table_read_error_v2(table_def, e)


@blocking_func(DATA_BACKEND_THREAD_POOL)
def _read_selected_parquet_sample(
    parquet_files: tuple[str, ...],
    column_names: tuple[str, ...],
    num_rows: int,
) -> pd.DataFrame:
    """Read a bounded sample from only the manifest-selected Parquet files."""
    frames = []
    remaining = num_rows
    for path in parquet_files:
        if remaining <= 0:
            break
        with fsutil.get_binary_handle(path, 'rb') as handle:
            parquet_file = pq.ParquetFile(handle)
            for batch in parquet_file.iter_batches(
                    batch_size=remaining, columns=list(column_names)):
                frame = batch.to_pandas().reindex(columns=column_names)
                frames.append(frame)
                remaining -= len(frame)
                if remaining <= 0:
                    break
    if not frames:
        return pd.DataFrame(columns=column_names)
    return stringify_sample_dataframe(pd.concat(frames, ignore_index=True))


async def _get_selected_sample_rows(
    table_def: TableDefinition,
    parquet_files: tuple[str, ...],
) -> Dict[str, List[Optional[str]]]:
    column_names = tuple(col.name for col in table_def.cols)
    sample = await _read_selected_parquet_sample.coroutine(
        parquet_files,
        column_names,
        DEFAULT_NUM_SAMPLE_ROWS,
    )
    return {name: sample[name].tolist() for name in column_names}


def _parse_col_stats(
    version: Optional[TableDataVersion],
    col_stats: Dict[BaseColumnConfig, Dict[ColStatType, Stats]],
    table_def: TableDefinition,
) -> ColStatsAtVersion:
    # Collect all data in dictionaries
    column_stats: Dict[str, ColumnStats] = {}

    # Get column metadata
    column_metadata = _get_column_metadata(table_def)

    approx = False
    rows_used = 0
    for base_column_config, col_stats_dict in col_stats.items():
        try:
            stype = column_metadata[base_column_config.name]['stype']
            assert isinstance(stype, Stype)
            dtype = column_metadata[base_column_config.name]['dtype']
            assert isinstance(dtype, Dtype)
            column_stats[base_column_config.name] = ColumnStats.from_stats_dict(
                stype, col_stats_dict, NUM_ROWS_FOR_APPROX_STATS, dtype)
            rows_used = column_stats[base_column_config.name].counts.total
            # Check approximation status of the returned stats, and if approx
            # check the number of rows used - currently will match across all
            # columns and all stats, but cleaner to retrieve from existing
            # returned structures then add to the function signatures.
            if column_stats[base_column_config.name].approx:
                approx = True
        except Exception as e:
            log.error(f"Error parsing stats {base_column_config} "
                      f"{col_stats_dict}")
            # The old wording sent the operator to review their own stype and
            # dtype, which reads as a configuration fault. A column can also
            # reach here correctly typed, with the statistics themselves
            # missing (kumo-ai/kumo#30917), so state what happened and leave
            # the cause to speak for itself.
            raise ValueError(f"Could not parse the statistics for column "
                             f"'{base_column_config.name}' (stype "
                             f"{base_column_config.stype}, dtype "
                             f"{base_column_config.dtype}): {e}") from e

    return ColStatsAtVersion(
        version=version,
        stats_dict=column_stats,
        approx=approx,
        num_rows_used=rows_used,
    )


async def _get_column_stats(
    table_def: TableDefinition,
    connector_spec: ConnectorSpec,
    col_stats_set: ColStatsSet = ColStatsSet.UI,
    check_refresh_status: bool = True,
) -> Tuple[ColStatsAtVersion, bool]:
    # Get column stat data
    log.info(f'Getting col stats {col_stats_set}: '
             f'{connector_spec.connector_id}.'
             f'{table_def.get_source_table_name()}')
    stats_request = col_stats_set.get_stats_types(table_def)
    table_data_version = await get_table_data_version(
        connector_spec, table_def.get_source_table_name())
    try:
        col_stats: Dict[BaseColumnConfig,
                        Dict[ColStatType,
                             Stats]] = await DataBackend.get_min_col_stats(
                                 connector_spec, table_def, stats_request,
                                 NUM_ROWS_FOR_APPROX_STATS,
                                 check_refresh_status=check_refresh_status,
                                 compute_missing_stats=False,
                                 allow_cached_approx_stats=True,
                                 table_data_version=table_data_version)
    except UnsupportedInDiskgraphError as exc:
        log.warning(
            "Column stats unavailable for %s.%s: %s",
            connector_spec.connector_id,
            table_def.get_source_table_name(),
            exc,
        )
        col_stats = {}

    statistics_available = col_stat_dict_complete(col_stats, stats_request)
    if not statistics_available:
        DataBackend.start_min_col_stats_cache_warmup(
            connector_spec,
            table_def,
            stats_request,
            NUM_ROWS_FOR_APPROX_STATS,
            check_refresh_status=check_refresh_status,
            table_data_version=table_data_version,
        )

    return (
        _parse_col_stats(version=None, col_stats=col_stats,
                         table_def=table_def),
        statistics_available,
    )


async def _get_col_stats_and_sample_rows(
    table_def: TableDefinition,
    connector_spec: ConnectorOpenSpec,
    col_stats_set: ColStatsSet = ColStatsSet.UI,
    check_refresh_status: bool = True,
) -> ColStatsResponse:
    # A Custom SQL table has no physical source table: its stats and sample
    # rows come from the ingested snapshot (post-materialization) or the
    # persisted Validate-time approx stats (pre-materialization), never from a
    # connector read of the synthetic alias.
    if table_def.is_sql:
        return await _get_sql_col_stats_and_sample_rows(table_def,
                                                        connector_spec,
                                                        col_stats_set)
    # Get sample rows
    column_sample_rows = await _get_sample_rows(table_def, connector_spec)

    # Get table column metadata
    column_metadata = _get_column_metadata(table_def)

    # Get column stats
    stats, statistics_available = await _get_column_stats(
        table_def, connector_spec, col_stats_set, check_refresh_status)

    # Build column summaries
    columns = _build_column_summaries(
        column_metadata=column_metadata,
        stats=stats,
        sample_rows=column_sample_rows,
    )

    return ColStatsResponse(statistics_available=statistics_available,
                            columns=columns, approx=stats.approx,
                            rows_used=stats.num_rows_used)


@connector_activity_registry.connector_authorized_activity
async def infer_table_metadata_activity(
    reference: ConnectorExecutionRef,
    request: table_api.TableMetadataRequest,
    reference_stypes: Optional[SqlReferenceStypes] = None,
) -> table_api.TableMetadataResponse:
    # ``reference_stypes`` is the Custom SQL stype-inheritance hints the driver
    # resolved from sibling Kumo tables (state DB) and threaded in; it is None
    # for physical tables and for a SQL query with no inheritable reference.
    connector = await get_connector_from_spec_async(
        reference, resolved_factory=get_connector_from_spec)
    async with async_connector_lifecycle(connector, open_connector=True):
        try:
            response, _, _ = await _resolve_table_metadata_with_connector(
                request,
                connector,
                connector_type=(reference_stypes.connector_type
                                if reference_stypes else None),
                reference_stypes=(reference_stypes.stypes
                                  if reference_stypes else None),
            )
            return response
        except (ConnectorError, ValueError) as error:
            source, _, _ = extract_source_table_type(request.source_table)
            if not isinstance(source, CustomSQLSourceTable):
                raise
            raise CustomSqlQueryError(
                message=_sql_validation_error_message(error)) from None


@connector_activity_registry.connector_authorized_activity
async def validate_table_definition_activity(
    reference: ConnectorExecutionRef,
    definition: TableDefinition,
    max_table_size_limits: Dict[str, int],
) -> None:
    connector = await get_connector_from_spec_async(
        reference, resolved_factory=get_connector_from_spec)
    async with async_connector_lifecycle(connector, open_connector=True):
        source_metadata = await _get_source_table_metadata.coroutine(
            str(definition.get_source_table_name()), connector)
        _validate_col_names_v2(definition)
        _validate_cols_are_unique_v2(definition)
        _validate_pkey_and_time_col_v2(definition)
        await _validate_table_metadata_v2.coroutine(definition, connector,
                                                    source_metadata)
        await _validate_table_size_v2.coroutine(definition, connector,
                                                max_table_size_limits)
        await _validate_table_permissions_v2.coroutine(definition, connector)


async def _resolve_custom_sql(
    reference: ConnectorExecutionRef,
    request: table_api.SqlTableResolveRequest,
    reference_stypes: Optional[SqlReferenceStypes] = None,
) -> table_api.SqlTableResolveResponse:
    try:
        connector, connector_type = await resolve_connector_and_type_async(
            reference)
        async with async_connector_lifecycle(connector, open_connector=True):
            return await _validate_and_preview_sql_table(
                request, connector, connector_type,
                reference_stypes=(reference_stypes.stypes
                                  if reference_stypes else None))
    except Exception as error:
        return table_api.SqlTableResolveResponse(
            errors=[_sql_resolve_error(error)])


@connector_activity_registry.connector_authorized_activity
async def validate_custom_sql(
    reference: ConnectorExecutionRef,
    request: table_api.SqlTableResolveRequest,
    reference_stypes: Optional[SqlReferenceStypes] = None,
) -> table_api.SqlTableResolveResponse:
    r"""Resolve Custom SQL inside the credential-broker activity."""
    return await _resolve_custom_sql(reference, request, reference_stypes)


@connector_activity_registry.connector_authorized_activity
async def get_table_col_stats_and_sample_rows_activity(
    reference: ConnectorExecutionRef,
    definition: TableDefinition,
) -> ColStatsResponse:
    from rest.backends.data_backend import connector_broker

    resolved = await connector_broker.connector_credential_broker.resolve_async(
        reference)
    return await _get_col_stats_and_sample_rows(definition, resolved)


@blocking_func(DATA_BACKEND_THREAD_POOL)
def _load_stats_from_per_job_dir(
        stats_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load column stats from file in a blocking manner.

    This function handles the blocking I/O operation of loading column
    statistics from a JSON file. The stats are stored in a specific
    format where:
    - Outer dict keys are formatted as:
        "{column_name}__{stype}__{stat_type}__{approx}".
    - Inner dict contains the actual statistics data that can be converted to
        Stats objects.

    Args:
        stats_path: Full path to the JSON file containing column statistics.

    Returns:
        Optional[Dict[str, Dict[str, Any]]]: A nested dictionary containing
            the column statistics where:
            - Outer key is the column identifier string
            - Inner dict contains the statistics data
        None: If the file cannot be loaded or parsed.
    """
    return load_json(stats_path, Dict[str, Dict[str, Any]])


@deprecated(reason="Obsolete after DB migration is complete")
def _build_column_summaries(
    column_metadata: Dict[str, Any],
    stats: ColStatsAtVersion,
    sample_rows: Optional[Dict[str, List[Any]]] = None,
) -> List[ColumnSummary]:
    """Build list of column summaries from metadata and stats.

    Args:
        column_metadata: Dictionary of column metadata
        stats: Column statistics
        sample_rows: Optional dictionary of sample rows per column
    """
    columns = []
    for data in column_metadata.values():
        col_name = data['name']
        assert isinstance(col_name, str)
        stype = data['stype']
        dtype = data['dtype']
        assert isinstance(stype, Stype)
        assert isinstance(dtype, Dtype)
        stype = stype.title_case()
        dtype = dtype.title_case()

        columns.append(
            ColumnSummary(
                name=col_name,
                stype=stype,
                dtype=dtype,
                sample_rows=[]
                if sample_rows is None else sample_rows[col_name],
                statistics=stats.stats_dict.get(col_name),
            ))
    return columns


async def try_load_col_stats_from_per_job_dir(
    table_def: TableDefinition,
    connector_spec: ConnectorSpec,
    col_stats_set: ColStatsSet,
    job_home_dir: str,
) -> Optional[ColStatsResponse]:
    """Try to load and parse column statistics from a job's output directory.

    Loads the raw stats file produced by a job, parses entries into
    ``Stats`` objects keyed by column config, and assembles a
    ``ColStatsResponse`` with metadata and sample rows.
    """
    stats_path = osp.join(job_home_dir, 'outputs', 'col_stats')
    if not fsutil.exists(stats_path):
        log.error(f"Column stats file not found at {stats_path}")
        return None

    raw_stats = await _load_stats_from_per_job_dir.coroutine(stats_path)
    if raw_stats is None:
        log.error(f"Failed to load column stats from file at {stats_path}")
        return None

    col_stats_types_dict = col_stats_set.get_stats_types(table_def)

    col_stats: Dict[BaseColumnConfig, Dict[ColStatType, Stats]] = {}
    for col_config, stat_types in col_stats_types_dict.items():
        col_stats[col_config] = {}
        for stat_type in stat_types:
            for approx in [True, False]:
                # Build the lookup key with the canonical builder so it matches
                # the persisted key byte-for-byte. A hand-rolled f-string with
                # bare ``{stat_type}`` renders ``ColStatType.MEAN`` (not
                # ``MEAN``) for a ``(str, Enum)`` on py3.12+, so it would never
                # match the ``.value``-built key and stats load empty.
                key = col_stat_key(col_config, stat_type, approx)
                if key in raw_stats:
                    col_stats[col_config][stat_type] = (Stats.from_config(
                        raw_stats[key]))

    statistics_available = col_stat_dict_complete(col_stats,
                                                  col_stats_types_dict)
    stats = (_parse_col_stats(None, col_stats, table_def)
             if statistics_available else ColStatsAtVersion(
                 version=None,
                 stats_dict={},
                 approx=False,
                 num_rows_used=0,
             ))
    column_metadata = _get_column_metadata(table_def)

    home = HomeDirManager(job_home_dir)
    source_name = str(table_def.get_source_table_name())
    selected_sample_files = None
    if source_name == TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME:
        sample_source = home.internal_transformed_predictions_file
    else:
        predictions = await resolve_prediction_artifact.coroutine(
            home, PredictionArtifactType.PREDICTIONS)
        sample_source = predictions.root if predictions is not None else None
        selected_sample_files = (predictions.files
                                 if predictions is not None else None)
    if sample_source is not None and fsutil.exists(sample_source):
        if selected_sample_files is not None:
            column_sample_rows = await _get_selected_sample_rows(
                table_def, selected_sample_files)
        else:
            column_sample_rows = await _get_sample_rows(
                table_def, connector_spec)
    else:
        # Direct customer outputs deliberately have no full internal Parquet
        # artifact. Statistics remain available; only preview rows are absent.
        column_sample_rows = None

    columns = _build_column_summaries(
        column_metadata=column_metadata,
        stats=stats,
        sample_rows=column_sample_rows,
    )
    return ColStatsResponse(
        statistics_available=statistics_available,
        columns=columns,
        approx=stats.approx,
        rows_used=stats.num_rows_used,
    )
