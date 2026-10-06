# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import field
from hashlib import md5
from typing import Dict, List, Optional

from kumo_core.types import TableStatType
from kumo_core.types.schema import (
    TS_FORMAT_METADATA,
    SourceTableID,
    TableFingerprint,
    VersionedSourceTableID,
)
from kumoapi.typing import Dtype, Stype
from temporalio import activity

from common_utils.kumo_timing import kumo_timing
from io_utils import fsutil, kumo_dataclass, logging
from io_utils.json import save_json
from kumo.connector.file_connector import FileConnector
from kumo.connector.timestamp_infer import map_timestamp_unit
from kumo.data.backend_cache import (
    DirectDiskgraphSourceDecision,
    TableDataTaskResult,
    TableRefreshStage,
)
from kumo.data.cache.graph_cache import GraphVersion
from kumo.validation import ValidationResponse
from kumo.validation.validation import ValidationError, ValidationException
from rest.backends.data_backend import connector_activity_registry
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.utils import (
    data_backend_cache,
    get_table_data_version,
)
from rest.backends.data_backend.validation import (
    ColStatsValidator,
    GraphRefreshValidator,
    IngestionValidator,
)
from rest.config.artifact_export import DatabricksManagedVolumeStage
from rest.config.batch_prediction_job import BatchPredictionJobInput
from rest.config.data_backend import (
    ConnectorExecutionRef,
    ConnectorOpenSpec,
    GraphDataVersionInfo,
)
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.table_v2 import (
    TableDefinition,
    TimestampUnit,
    is_file_source_table,
    table_args_to_kumo_table,
)
from rest.utils.driver_common import get_connector_from_spec
from service_framework.deploy_env_config import DeployEnvConfig
from statedb.config.common import ConnectorID, TableName
from statedb.config.table import TableDataVersion
from temporalio_commons import is_cancelled_exception

from . import ingest_table_data_activity_v2 as ingest_activity

log = logging.getLogger(__name__)


@kumo_dataclass
class GraphTableStatsResult:
    """Return value of ``validate_graph_table_stats``.

    Bundles the ``ValidationResponse`` with the total source-table size so
    callers can pass it through to ``DiskgraphIngestWorkflow`` as an integer
    arg — avoiding a separate activity to re-read the same cache. The total
    is only populated for ``TableRefreshStage.INGEST_TABLE``; it is 0 for
    col-stats validation stages where size information is irrelevant.
    """
    validation: ValidationResponse
    total_source_size_bytes: int = 0
    source_size_bytes_by_table: Dict[str, int] = field(default_factory=dict)


class DataBackendActivities:
    @staticmethod
    @connector_activity_registry.connector_authorized_activity
    async def start_bp_stats_activity(
        ctx: DataBackendContext,
        job_input: BatchPredictionJobInput,
        use_transformed_output: bool = False,
        direct_databricks_output_index: int | None = None,
        direct_databricks_stage: DatabricksManagedVolumeStage | None = None,
    ) -> None:
        """Activity wrapper around
        :meth:`DataBackend.start_bp_stats_activity` — see that method for
        the full contract.

        ``DataBackend`` is imported lazily because module ``data_backend``
        in turn imports the workflows module, which references this
        activities class — a top-level import would form a cycle.
        """
        from rest.backends.data_backend.data_backend import DataBackend
        return await DataBackend.start_bp_stats_activity(
            ctx,
            job_input,
            use_transformed_output,
            direct_databricks_output_index,
            direct_databricks_stage,
        )

    @staticmethod
    @connector_activity_registry.connector_authorized_activity
    async def compute_standalone_table_col_stats(
        connector_spec: ConnectorExecutionRef,
        table: TableDefinition,
        data_version: TableDataVersion,
    ) -> bool:
        """Activity wrapper for standalone table refresh col-stats."""
        from rest.backends.data_backend.data_backend import DataBackend
        return await DataBackend.compute_standalone_table_col_stats(
            connector_spec, table, data_version)

    @staticmethod
    @activity.defn
    async def wait_on_workflow(
        ctx: DataBackendContext,
        wf_id: str,
        run_id: Optional[str] = None,
    ):
        """
        Generic activity to wait and get result of a running or completed
        execution of workflow id=`wf_id`.

        Args:
            ctx: Data backend context
            wf_id: Workflow ID to wait on
            run_id: Exact workflow execution to wait on, when known
        """
        with kumo_timing("data_backend.wait_on_workflow", wf_id=wf_id):
            return await ctx.temporal_client().result(wf_id, run_id=run_id)

    # Sync (threadpool, not event loop): the prologue takes the process-global
    # TableVersionCache lock; waiting on it on the loop freezes the worker.
    @staticmethod
    @connector_activity_registry.connector_authorized_activity
    def update_table_timestamp_units(
        table_def: TableDefinition,
        connector_spec: ConnectorExecutionRef,
        base_tables_args_list: Optional[List[TableDefinition]],
        base_tables_connector: Optional[ConnectorExecutionRef],
    ) -> TableDefinition:
        # Retained for compatibility with serialized activity payloads.
        del base_tables_args_list, base_tables_connector
        connector = None
        try:
            connector = get_connector_from_spec(connector_spec)

            kumo_table = table_args_to_kumo_table(table_def, connector)
            native_temporal_cols = set()
            if any(
                    col.stype == Stype.timestamp
                    and col.timestamp_format is None
                    for col in table_def.cols):
                source_metadata = None
                try:
                    if table_def.is_sql:
                        source_metadata = connector.get_query_metadata(
                            table_def.source_table.sql)
                    else:
                        source_metadata_list = connector.get_table_metadata(
                            [table_def.get_source_table_name()],
                            skip_cache=True,
                            skip_inferral=True,
                        )
                        if source_metadata_list:
                            source_metadata = source_metadata_list[0]
                    source_cols = (source_metadata.all_cols()
                                   if source_metadata is not None else [])
                    if not source_cols:
                        log.warning(
                            'Source schema metadata was empty; falling back '
                            'to timestamp value inference.')
                    else:
                        native_temporal_cols = {
                            col.name for col in source_cols
                            if col.dtype in {Dtype.date, Dtype.time}
                        }
                except Exception as e:
                    if is_cancelled_exception(e):
                        raise
                    log.warning(
                        'Source schema metadata lookup failed '
                        '(error_type=%s); falling back to timestamp value '
                        'inference.', type(e).__name__)
            if native_temporal_cols:
                cols_to_new_dtypes = (
                    kumo_table._infer_dtypes_from_stypes_blocking(
                        native_temporal_cols))
            else:
                cols_to_new_dtypes = (
                    kumo_table._infer_dtypes_from_stypes_blocking())
            for col in kumo_table.cols:
                if col.name in cols_to_new_dtypes:
                    col.dtype = cols_to_new_dtypes[col.name]
            kumo_table_cols = {col.name: col for col in kumo_table.data_cols}

            # Build a new TableDefinition with the resolved hint persisted back
            # onto the frozen v2 Column. Column is frozen, so we construct new
            # instances instead of mutating. Inference is authoritative and
            # forwards BOTH sides of the hint: a UNIT hint is written as a
            # ``TimestampUnit`` enum, and a real strftime FORMAT hint (from
            # ``metadata['TS_FORMAT']``) is written as a raw string. The v2
            # ``timestamp_format`` field overloads on type (enum = unit,
            # str = format).
            from rest.config.table_v2 import Column
            new_cols = []
            for col in table_def.cols:
                inferred = kumo_table_cols.get(col.name)
                ts_format_value = None
                if inferred is not None:
                    unit = map_timestamp_unit(inferred.timestamp_unit)
                    if unit is not None:
                        ts_format_value = TimestampUnit(unit)
                    else:
                        fmt = inferred.metadata.get(TS_FORMAT_METADATA)
                        if fmt:  # a non-empty strftime format
                            ts_format_value = fmt
                if ts_format_value is not None:
                    new_cols.append(
                        Column(
                            name=col.name, stype=col.stype, dtype=col.dtype,
                            timestamp_format=ts_format_value,
                            multicategorical_delimiter=(
                                col.multicategorical_delimiter)))
                else:
                    new_cols.append(col)
            updated = TableDefinition(
                cols=new_cols,
                source_table=table_def.source_table,
                pkey=table_def.pkey,
                time_col=table_def.time_col,
                end_time_col=table_def.end_time_col,
            )
            log.info('Updated table timestamp units')
            return updated
        except Exception as e:
            if is_cancelled_exception(e):
                raise
            ingest_activity._raise_sanitized_table_ingestion_error(e)
        finally:
            if connector is not None:
                connector.close()

    @staticmethod
    @activity.defn
    def validate_graph_table_stats(
        ctx: DataBackendContext,
        versioned_table_dict: Dict[VersionedSourceTableID, TableDefinition],
        stage: TableRefreshStage,
        connector_specs: List[ConnectorExecutionRef],
        aliases: Dict[VersionedSourceTableID, List[TableName]],
    ) -> GraphTableStatsResult:
        validator = GraphRefreshValidator(ctx._config)
        if stage == TableRefreshStage.INGEST_TABLE:
            validation = validator.post_table_ingest(ctx, versioned_table_dict,
                                                     connector_specs)
            # Sum SOURCE_SIZE for pool admission from validated, cached stats.
            stats = data_backend_cache().get_many_min_exact_stats(
                [(k, v.fingerprint()) for k, v in versioned_table_dict.items()],
                skip_col_stats=True)
            source_sizes, non_positive = {}, []
            for table_id, table, stat in zip(versioned_table_dict.keys(),
                                             versioned_table_dict.values(),
                                             stats):
                if stat and TableStatType.SOURCE_SIZE in stat.table_stats:
                    source_size = stat.table_stats[TableStatType.SOURCE_SIZE]
                    size = int(source_size.value)
                else:
                    cached = data_backend_cache().get_table_stats(
                        table.fingerprint(),
                        [TableStatType.SOURCE_SIZE],
                        table_id.version,
                    )
                    source_size = cached.get(TableStatType.SOURCE_SIZE)
                    size = int(source_size.value) if source_size else 0
                if size <= 0:
                    non_positive.append(str(table_id.source_table_name))
                else:
                    for alias in aliases[table_id]:
                        source_sizes[str(alias)] = size
            _raise_for_unusable_source_sizes(non_positive, validation)
            if non_positive:
                return GraphTableStatsResult(validation=validation)
            total = sum(source_sizes.values())
            return GraphTableStatsResult(
                validation=validation, total_source_size_bytes=total,
                source_size_bytes_by_table=source_sizes)
        else:
            assert (stage == TableRefreshStage.COMPUTE_MIN_COL_STATS
                    or stage == TableRefreshStage.COMPLETED)
            validation_aliases = {
                table_id: names[0]
                for table_id, names in aliases.items()
            }
            return GraphTableStatsResult(validation=validator.post_col_stats(
                versioned_table_dict, aliases=validation_aliases))

    @staticmethod
    @connector_activity_registry.connector_authorized_activity
    def validate_table_ingest(
        table_args: TableDefinition,
        data_version: TableDataVersion,
        deploy_config: DeployEnvConfig,
        existing_errors: Optional[List[ValidationError]] = None,
        connector_spec: Optional[ConnectorExecutionRef] = None,
    ) -> ValidationResponse:
        """Per-table post-ingest validator dispatch.

        Emits the three user-facing post-ingest checks: empty-table error
        (``n_rows == 0``), "<10 rows" small-table warning, and per-table
        size-cap check (via ``DataScaleLimits.validate_table``).

        ``connector_spec`` is forwarded so the validator can opt out of the
        local-path checks for a table DiskGraph reads directly, which stages
        no Kumo Parquet for either of them to read. When ``connector_spec`` is
        ``None`` we default to ``check_local=True`` (FileConnector path).
        """
        connector = None
        try:
            check_local = True
            require_materialization_stage = True
            if connector_spec is not None:
                source_table_name = table_args.get_source_table_name()
                decision = None
                if is_file_source_table(table_args):
                    decision = (data_backend_cache().
                                direct_diskgraph_source_decision.get(
                                    table_args.fingerprint(),
                                    str(data_version),
                                    cls=DirectDiskgraphSourceDecision,
                                ))
                if decision is not None:
                    direct_diskgraph_source = decision.source is not None
                    check_local = require_materialization_stage = (
                        not direct_diskgraph_source)
                else:
                    connector = get_connector_from_spec(connector_spec)
                    direct_diskgraph_source = (
                        isinstance(connector, FileConnector)
                        and connector.uses_direct_diskgraph_source_for(
                            source_table_name))
                    check_local = not direct_diskgraph_source
                    require_materialization_stage = not direct_diskgraph_source
            validator = IngestionValidator(deploy_config)
            return validator.post_ingestion(
                table_args,
                data_version,
                existing_errors=existing_errors,
                check_local=check_local,
                require_materialization_stage=require_materialization_stage,
            )
        finally:
            if connector is not None:
                connector.close()

    @staticmethod
    @activity.defn
    def validate_table_col_stats(
        connector_id: str,
        table_args: TableDefinition,
        data_version: TableDataVersion,
        pkey_duplicate_count: Optional[int] = None,
    ) -> ValidationResponse:
        """Per-table col-stats validator dispatch.

        Emits the ``"Duplicate primary keys"`` /
        ``"Missing or invalid primary keys"`` /
        ``"Large number invalid entries"`` warnings that the UI and
        ``test/rest/integ/test_schema.py`` rely on.
        """
        cache = data_backend_cache()
        fingerprint = table_args.fingerprint()
        cache.load_col_stats_cache(fingerprint, data_version)

        validator = ColStatsValidator()
        return validator.post_col_stats(
            connector_id,
            table_args,
            data_version,
            pkey_duplicate_count=pkey_duplicate_count,
        )

    @staticmethod
    @activity.defn
    def validate_graph_edge_stats(
            input: GraphDataComputeInput,
            graph_version_info: GraphDataVersionInfo,
            deploy_config: DeployEnvConfig) -> ValidationResponse:
        return GraphRefreshValidator(deploy_config).post_edge_stats(
            input, graph_version_info)

    @staticmethod
    @activity.defn
    def create_table_cache_dir(
        connector_id: str,
        source_table_name: str,
        data_version: str,
        table_fingerprint: TableFingerprint,
    ) -> str:
        persist_dir = data_backend_cache().create_table_cache_dir(
            connector_id, source_table_name, data_version, table_fingerprint)
        return persist_dir._table_dir

    @staticmethod
    @activity.defn
    async def save_table_version_activity(
        job_id: str,
        connector_id: str,
        source_table_name: str,
        data_version: str,
        is_ingestion_successful: Optional[bool] = True,
    ):
        """
        Globally, the version is saved under <environment>/data-persist-dir/data_backend_cache/versions/latest/<connector_name>/<table_name>/  # noqa
        if the workflow is successful, the version is saved in a file called
        `version`. If the workflow fails, it is saved in a file called
        `failure_version`. For fetching table metadata, always the successful
        `version` is expected to be used, for fetching recent ingestion failures, the
        `failure_version` is expected to be used.
        Args:
        job_id (str): Id associated with the job requesting the save table version.
        connector_id (str): Id of the Connector associated with the table whose
            version is being saved.
        source_table_name (str): Source table name whose version is being saved.
        data_version (str): Version string required to be saved.
        refresh_table_version (bool): If true, update the table version in the
            backend cache
        is_ingestion_successful (bool): If the upstream ingestion workflow for this
            version was successful.
        """
        if not is_ingestion_successful:
            # Save table version under failure file
            await data_backend_cache().table_version_cache.save.coroutine(
                connector_id,
                source_table_name,
                data_version,
                is_version_ingestion_successful=False,
            )
            return

        await data_backend_cache().table_version_cache.save.coroutine(
            connector_id, source_table_name, data_version)

        # Save the table data version for a given job_id (training job, batch
        # prediction job, graph stats job). This helps to make sure that we are
        # consistently referring to the same version of a table during the
        # entire span of the given job.
        if job_id:
            await data_backend_cache(
            ).table_version_cache.save_for_job.coroutine(  # noqa: E501
                job_id, connector_id, source_table_name, data_version)

    @staticmethod
    @activity.defn
    async def save_source_table_ingest_result(
            source_table_id: SourceTableID,
            table_data_version: TableDataVersion,
            task_result: TableDataTaskResult):
        await data_backend_cache().table_refresh_result.save.coroutine(
            source_table_id.connector_name, source_table_id.source_table_name,
            table_data_version, task_result)

    @staticmethod
    @activity.defn
    async def save_table_refresh_result(
        table: TableDefinition,
        data_version: TableDataVersion,
        task_result: TableDataTaskResult,
    ):
        # DIAGNOSTIC INSTRUMENTATION (PR #28405 hang investigation).
        log.info(
            "ACTIVITY_SAVE_TABLE_REFRESH_RESULT_BEGIN table=%s stage=%s "
            "data_version=%s cache_key=%s",
            table.get_source_table_name(),
            task_result.stage,
            data_version,
            table.cache_key(),
        )
        # TODO: this is a bit hacky and should be revisited when we redesign
        # the data backend and backend cache.
        #
        # Data backend cache does not distinguish source table from kumo table
        # while there could be multiple kumo tables that point to the same
        # source table. Because each kumo table has its own column args, the
        # status/result of table refresh workflow (including col stats) is
        # specific to each kumo table.  Therefore we construct a unique hash
        # key to represent a unique definition of kumo table
        await data_backend_cache().table_refresh_result.save.coroutine(
            table.get_connector_id(), table.cache_key(), data_version,
            task_result)
        log.info(
            "ACTIVITY_SAVE_TABLE_REFRESH_RESULT_END table=%s stage=%s",
            table.get_source_table_name(),
            task_result.stage,
        )

    @staticmethod
    @activity.defn
    async def save_colstat_task_result(
        table: TableDefinition,
        data_version: TableDataVersion,
        task_result: TableDataTaskResult,
    ):
        assert task_result.stage in [
            TableRefreshStage.COMPUTE_FULL_COL_STATS,
            TableRefreshStage.COMPUTE_MIN_COL_STATS
        ]
        # create_table_cache_dir is a cheap makedirs(exist_ok=True); the
        # write is offloaded off the activity thread pool via save_json.coroutine.
        table_dir = data_backend_cache().create_table_cache_dir(
            table.get_connector_id(), table.cache_key(), data_version,
            table.fingerprint())
        file_path = table_dir.colstat_task_result(task_result.stage)
        log.debug('Save %s task result to %s', task_result.stage, file_path)
        await save_json.coroutine(task_result, file_path)

    @staticmethod
    @activity.defn
    async def save_graph_data_version(graph_input: GraphDataComputeInput,
                                      version_info: GraphDataVersionInfo,
                                      job_id: Optional[str] = None):
        cache = data_backend_cache()
        # Persist graph version -> {table: table_ver} mapping
        await save_json.coroutine(
            version_info,
            cache.graph_cache.version_info_path(
                graph_input.graph_key, version_info.graph_data_version))

        # Persist job_id -> {table: table_ver} mapping
        if job_id:
            table_versions = version_info.table_data_versions
            for table_id, table in (list(graph_input.tables.items()) +
                                    list(graph_input.base_tables.items())):
                alias = graph_input.table_aliases[table_id]
                table_version = table_versions[alias]
                (await cache.table_version_cache.save_for_job.coroutine(
                    job_id, table.get_connector_id(),
                    table.get_source_table_name(), table_version))

    @staticmethod
    @activity.defn
    async def copy_colstat_to_job_dir(
        input_path: str,
        output_path: str,
    ) -> None:
        r"""Copy column statistics from cache to a job-specific directory.

        This activity is used to persist column statistics for a specific
        job, typically called after the col-stats validator dispatch. The
        stats are copied from the general cache location to a job-specific
        directory so that even when stats in cache are deleted, the stats
        can still be retrieved from the job-specific directory by ui.

        Args:
            input_path (str): Source path where the column stats are cached.
            output_path (str): Destination path in the job directory where
                stats should be copied.
        """
        await fsutil.copy.coroutine(input_path, output_path)
        log.info(f"Successfully copied bp stats to {output_path}")

    @staticmethod
    @connector_activity_registry.connector_authorized_activity
    async def refresh_graph_data_version(
            graph_input: GraphDataComputeInput,
            job_id: Optional[str] = None) -> GraphDataVersionInfo:
        return await DataBackendActivities._refresh_graph_data_version(
            graph_input,
            graph_input.connectors,
            job_id,
        )

    @staticmethod
    @connector_activity_registry.connector_authorized_activity
    async def get_connector_table_data_version(
        connector: ConnectorExecutionRef,
        source_table_name: str,
        refresh_table_version: bool = False,
        table_def: Optional[TableDefinition] = None,
    ) -> TableDataVersion:
        # ``table_def`` selects the Custom SQL content-hash version branch (SQL
        # + referenced-table versions + output cols); physical tables pass None
        # and take the connector-probe path.
        return await get_table_data_version(
            connector,
            source_table_name,
            refresh_table_version=refresh_table_version,
            table_def=table_def,
        )

    @staticmethod
    async def _refresh_graph_data_version(
        graph_input: GraphDataComputeInput,
        connector_specs: Dict[ConnectorID, ConnectorOpenSpec],
        job_id: Optional[str],
    ) -> GraphDataVersionInfo:
        """
        Refresh data version for all tables in the graph.  Then calculate
        graph data version as hash of concantenated table versions, and persist
        it as latest version in persist dir.
        If job_id is specified, all refreshed table data versions will be saved
        for the job_id. v2 graphs do not contain views.
        """
        log.info('Refresh data version for graph: %s', graph_input)

        # TODO(jinu): What if two tables have same alias but different
        # connector_id?
        table_version_coros = {}
        for table_id, table in (list(graph_input.tables.items()) +
                                list(graph_input.base_tables.items())):
            alias = graph_input.table_aliases[table_id]
            pinned = graph_input.pinned_data_versions.get(table_id)
            if pinned is not None:
                version_coro = asyncio.Future[TableDataVersion]()
                version_coro.set_result(pinned)
            else:
                # A Custom SQL table's version hashes the SQL + referenced-table
                # versions + output cols (``table_def`` selects that branch),
                # not a connector probe of the synthetic alias.
                version_coro = get_table_data_version(
                    connector_spec=connector_specs[table.get_connector_id()],
                    source_table_name=table.get_source_table_name(),
                    refresh_table_version=True, table_def=table)
            table_version_coros[alias] = version_coro

        table_version_values: List[TableDataVersion] = await asyncio.gather(
            *table_version_coros.values())
        table_versions = dict(
            zip(table_version_coros.keys(), table_version_values))

        md5_hasher = md5()
        for table_name, table_version in sorted(table_versions.items()):
            md5_hasher.update(f'{table_name}={table_version}'.encode())
        graph_data_version = md5_hasher.hexdigest()

        # Save graph and table version info.
        await data_backend_cache().graph_cache.save_latest_graph_version.\
            coroutine(graph_input.graph_key, graph_data_version)

        version_info = GraphDataVersionInfo(GraphVersion(graph_data_version),
                                            table_versions)
        await DataBackendActivities.save_graph_data_version(
            graph_input, version_info, job_id)
        return version_info


def _raise_for_unusable_source_sizes(non_positive: List[str],
                                     validation: ValidationResponse) -> None:
    if non_positive and validation.ok:
        raise ValidationException(
            title="Table size could not be determined", message=(
                f"Kumo recorded a source size of zero for "
                f"{', '.join(non_positive)}, so it cannot estimate the local "
                f"disk this graph needs. Re-ingest the affected tables."))
