# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import time
from datetime import datetime
from typing import Dict, List, Optional

import tenacity
from kumo_core.diskgraph.types import InputDataStats
from kumo_core.types import ColStatType, TableStatType
from kumo_core.types.schema import BaseColumnConfig
from kumo_core.version import get_kumo_version

from async_utils.sync import blocking_func
from io_utils import logging
from kumo.config import GraphConfig
from kumo.data import get_active_cache
from kumo.data.backend_cache import TableIngestionInfo
from kumo.data.cache.graph_cache import VersionedGraphKey
from kumo.data.input_data_stats_helpers import compute_edge_counts
from kumo.diskgraph import diskgraph_native_tt_artifacts
from kumo.graph import Table
from kumo.graph.column import Stats
from kumo.graph.graph import Graph
from kumo.stats import is_missing_stat_value
from rest.backends.ml_backend.ml_job_output_store import MLJobOutputStore
from rest.config.batch_prediction_job import BatchPredictionJobOutput
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.ml_backend import (
    MaterializationTaskInfo,
    MaterializedColumnInfo,
    MaterializedTableInfo,
    TimeColInfo,
)
from rest.config.training_job import TrainingJobOutput
from rest.settings import DataScaleLimits, get_db_storage_root
from service_framework.deploy_env_config import DeployEnvConfig, deploy_env
from statedb.config.common import (
    BatchPredictionJobID,
    ConnectorID,
    GraphID,
    JobEventLogEntry,
    JobID,
    JobStatus,
    JobType,
    PredictiveQueryID,
    TrainingJobID,
    now_utc,
    to_table_id,
)
from statedb.db import MLJobService, open_state_db_from_deploy_config

logger = logging.getLogger(__name__)


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=2),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def update_job_status(
    job_id: JobID,
    job_type: JobType,
    job_output_store: MLJobOutputStore,
    log_entry: JobEventLogEntry,
):
    if job_type == JobType.TRAINING_JOB:
        job_output = TrainingJobOutput(status=JobStatus.RUNNING, log=log_entry)
        await job_output_store.save_training_job_output(TrainingJobID(job_id),
                                                        job_output)
    elif job_type == JobType.BATCH_PREDICTION_JOB:
        job_output = BatchPredictionJobOutput(status=JobStatus.RUNNING,
                                              log=log_entry)
        await job_output_store.save_batch_prediction_job_output(
            BatchPredictionJobID(job_id), job_output)
    else:
        raise ValueError(
            f"Received JobID: {job_id} with unsupported JobType: {job_type}")


def make_materialization_info(
        start_time: datetime, graph: Graph, job_id: JobID, graph_id: GraphID,
        predictive_query_id: PredictiveQueryID,
        temporal_workflow_id: str) -> MaterializationTaskInfo:
    """Collect information about the materialization task, for the purpose of
    storing on S3 to subsequently show in the UI. This is intended to provide a
    source of truth "snapshot" about what data was used to train a particular
    query or batch prediction job, and it is very useful for ML Ops.
    """
    return MaterializationTaskInfo(
        table_stats=_make_materialized_table_info(graph),
        code_version=get_kumo_version(),
        start_time=start_time,
        end_time=now_utc(),
        job_id=job_id,
        graph_id=graph_id,
        predictive_query_id=predictive_query_id,
        temporal_workflow_id=temporal_workflow_id,
    )


def _make_materialized_table_info(graph: Graph) -> List[MaterializedTableInfo]:
    output: List[MaterializedTableInfo] = []
    data_backend_cache = get_active_cache()
    data_backend_cache.load_all_table_caches(graph.config())

    for table in graph.tables:
        assert table.data_source is not None

        col_stats = data_backend_cache.get_col_stats(
            table.fingerprint(), stats_dict=_col_info_args(table),
            data_version=table.data_version, approx=False)
        required_stats = [TableStatType.SOURCE_SIZE, TableStatType.NUM_ROWS]
        table_stats = data_backend_cache.get_table_stats(
            table.fingerprint(), stats_list=required_stats,
            data_version=table.data_version)

        # Retry with cache reload if stats are missing (can happen during
        # concurrent graph refresh). See #24175 / #24050.
        missing_stats = [s for s in required_stats if s not in table_stats]
        if missing_stats:
            logger.warning(
                f"Table stats {missing_stats} missing for table "
                f"'{table.source_name}' (data_version="
                f"{table.data_version}). Reloading cache and retrying.")
            time.sleep(2)
            data_backend_cache.load_all_table_caches(graph.config())
            table_stats = data_backend_cache.get_table_stats(
                table.fingerprint(), stats_list=required_stats,
                data_version=table.data_version)
            missing_stats = [s for s in required_stats if s not in table_stats]

        logger.info("col_stats loaded for table (for MaterializedTableInfo):"
                    f" {table.source_name}")
        for key, value in col_stats.items():
            logger.info(
                f"table {table.source_name} col: {key} stats: {value.keys()}")
        logger.info("table_stats loaded for table (for MaterializedTableInfo):"
                    f" {table.source_name}\n{table_stats}")

        ingest_info = data_backend_cache.ingestion_info.get_v1(
            table.data_source, table.source_name, table.data_version,
            TableIngestionInfo)
        if ingest_info is None:
            # Ingest info may not be available for old data in the data cache
            # TODO(zack): remove this once we have a mechanism to force
            # reingestion
            logger.warning(
                f"Ingestion info is not available for table '{table.name}'."
                "Please force re-ingestion for all of the tables.")
            continue

        if missing_stats:
            logger.error(
                f"Table stats {missing_stats} still missing for table "
                f"'{table.source_name}' after retry. Using -1 as fallback.")

        source_size = int(table_stats[TableStatType.SOURCE_SIZE].value
                          ) if TableStatType.SOURCE_SIZE in table_stats else -1
        num_rows = int(table_stats[TableStatType.NUM_ROWS].value
                       ) if TableStatType.NUM_ROWS in table_stats else -1

        output.append(
            MaterializedTableInfo(
                size_bytes_source=source_size,
                ingest_start_time=ingest_info.ingest_start_time,
                ingest_end_time=ingest_info.ingest_end_time,
                num_rows=num_rows,
                data_version=table.data_version,
                columns=_make_materialized_cols_info(table, col_stats),
                pkey=_get_pkey_name(table),
                time_col=_get_time_col_info(table, col_stats),
                table_id=to_table_id(table.name),
                table_name=to_table_id(table.name).table_name,
                connector_id=ConnectorID(table.data_source),
                source_table_name=table.source_name,
                code_version=ingest_info.code_version,
                temporal_workflow_id=ingest_info.temporal_workflow_id,
            ))
        logger.info(f"MaterializedTableInfo created for {table.name}")

    return output


def _get_pkey_name(table: Table) -> Optional[str]:
    if table.pkey is None:
        return None
    return table.pkey.name


def _get_time_col_info(
    table: Table,
    col_stats: Dict[BaseColumnConfig, Dict[ColStatType, Stats]],
) -> Optional[TimeColInfo]:
    if table.time_col is None:
        return None
    time_col_name = table.time_col.name

    if table.time_col.base_config() not in col_stats:
        raise AssertionError(
            f"Stats for time col '{time_col_name}' could not be found.")

    stats = col_stats[table.time_col.base_config()]

    return TimeColInfo(name=table.time_col.name,
                       min=_get_time_col_stat(ColStatType.MIN, stats),
                       max=_get_time_col_stat(ColStatType.MAX, stats))


def _get_time_col_stat(
    stat: ColStatType,
    col_stats: Dict[ColStatType, Stats],
) -> Optional[datetime]:
    output = col_stats[stat].value
    if is_missing_stat_value(output):
        return None
    # TT-NG predict col_stats carry epoch-int time stats; no-op for datetime.
    output = diskgraph_native_tt_artifacts.normalize_timestamp_like_value(
        output)
    if not isinstance(output, datetime):
        raise AssertionError(f"Data type for time column stat {output} "
                             f"must be datetime but got {type(output)}")
    return output


# The stats requested here must also be included in
# Column.required_stats so that a prediction task graph refresh
# computes them


def _col_info_args(table: Table) -> Dict[BaseColumnConfig, List[ColStatType]]:
    output: Dict[BaseColumnConfig, List[ColStatType]] = {}
    for col in table:
        output[col.base_config()] = [
            ColStatType.NUM_NA,
            ColStatType.MIN,
            ColStatType.MAX,
        ]
    return output


def _make_materialized_cols_info(
    table: Table, col_stats: Dict[BaseColumnConfig, Dict[ColStatType, Stats]]
) -> List[MaterializedColumnInfo]:
    output: List[MaterializedColumnInfo] = []
    for col in table:
        assert col.dtype is not None
        assert col.stype is not None
        stats = col_stats[col.base_config()]
        output.append(
            MaterializedColumnInfo(
                name=col.name,
                dtype=col.dtype,
                stype=col.stype,
                num_empty=int(stats[ColStatType.NUM_NA].value),
            ))
    return output


@blocking_func()
def estimate_input_data_stats(
        graph: Graph, key: VersionedGraphKey,
        deploy_config: Optional[DeployEnvConfig] = None) -> InputDataStats:
    """Estimate input data stats for the given graph.

    This function calculates input data stats — number of tables, number
    of edges, and the graph size — and estimates the materialized table
    size from the table's source size.

    Args:
        graph: The graph to estimate input data stats for.
        key: Versioned graph key to retrieve the pre-computed edge stats.
        deploy_config: The deploy environment configuration (used for
        data scale limits).

    Returns:
        An InputDataStats object containing the estimated input data stats.
    """
    for table in graph.tables:
        if table.data_version is None:
            raise ValueError(
                'table data versions must be resolved before input data '
                f'stats estimation; missing version for {table.name!r}')

    result = InputDataStats()
    backend_cache = get_active_cache()
    graph_edge_stats = backend_cache.graph_cache.get_graph_edge_stats(key)
    compute_edge_counts(graph, graph_edge_stats, result)

    total_table_size = 0
    # Use MATERIALIZED_SIZE if available from a previous run (accurate), or
    # fall back to SOURCE_SIZE (ingested parquet size). SOURCE_SIZE can
    # significantly underestimate the materialized graph size because SST
    # encoding, feature expansion, and edge materialization inflate it
    # (observed 2-3x for Databricks views). A safety multiplier is applied
    # to SOURCE_SIZE to compensate.
    SOURCE_SIZE_TO_MATERIALIZED_MULTIPLIER = 3.0
    max_retries = 3
    retry_delay_seconds = 2
    used_materialized_size = False

    data_scale_limits = (DataScaleLimits.parse_from_env(deploy_config)
                         if deploy_config else DataScaleLimits())
    max_graph_size = data_scale_limits.max_graph_total_size_bytes

    for retry_attempt in range(max_retries):
        missing_tables = []
        total_table_size = 0
        used_materialized_size = True

        for table in graph.tables:
            table_stats = backend_cache.get_table_stats(
                table.fingerprint(), stats_list=[
                    TableStatType.MATERIALIZED_SIZE,
                    TableStatType.SOURCE_SIZE,
                ], data_version=table.data_version)

            if TableStatType.MATERIALIZED_SIZE in table_stats:
                total_table_size += int(
                    table_stats[TableStatType.MATERIALIZED_SIZE].value)
            elif TableStatType.SOURCE_SIZE in table_stats:
                used_materialized_size = False
                total_table_size += int(
                    table_stats[TableStatType.SOURCE_SIZE].value)
            else:
                used_materialized_size = False
                missing_tables.append(table)
                logger.warning(
                    f"Neither MATERIALIZED_SIZE nor SOURCE_SIZE available "
                    f"for table {table.name} "
                    f"(data_version={table.data_version}) on attempt "
                    f"{retry_attempt + 1}/{max_retries}")

        # If all stats are available, break out of retry loop
        if not missing_tables:
            break

        # If this is not the last attempt, reload cache and retry
        if retry_attempt < max_retries - 1:
            logger.info(f"Retrying to load table size stats for "
                        f"{len(missing_tables)} table(s) after "
                        f"{retry_delay_seconds}s delay...")
            time.sleep(retry_delay_seconds)
        else:
            # Last attempt failed - log error and use 0 as fallback
            logger.error(
                f"Failed to load table size stats for "
                f"{len(missing_tables)} table(s) after {max_retries} "
                f"attempts. This should have been computed during "
                f"GraphRefresh. Using max graph size {max_graph_size} as "
                f"fallback. Tables missing stats: "
                f"{[t.name for t in missing_tables]}")
            # use max_graph_size to avoid the fallback value triggering
            # validation errors for being too large
            total_table_size = max_graph_size

    # When using SOURCE_SIZE (no MATERIALIZED_SIZE available), apply a
    # safety multiplier to account for the expansion from raw ingested
    # parquet to materialized SST files + edge tables. SST encoding,
    # feature expansion, reverse edges, and index overhead typically
    # inflate the size by 2-3x.
    if not used_materialized_size and total_table_size < max_graph_size:
        original_size = total_table_size
        total_table_size = int(total_table_size *
                               SOURCE_SIZE_TO_MATERIALIZED_MULTIPLIER)
        logger.info(f"Applied {SOURCE_SIZE_TO_MATERIALIZED_MULTIPLIER}x safety "
                    f"multiplier to SOURCE_SIZE estimate: {original_size} -> "
                    f"{total_table_size} bytes")

    # result already has edge counts from compute_edge_counts above.
    result.graph_size_bytes = total_table_size
    # Number of table nodes is not used downstream here; the number of
    # tables is used instead. So we set it to 0.
    for table in graph.tables:
        result.table_num_nodes[table.name] = 0
    return result


async def _open_ml_job_service() -> MLJobService:
    state_db = await open_state_db_from_deploy_config(
        deploy_env(),
        state_root=get_db_storage_root(),
    )
    return MLJobService(state_db)


async def _save_input_data_stats_to_state_db_async(
    job_id: str,
    input_data_stats: InputDataStats,
) -> None:
    service = await _open_ml_job_service()
    await service.save_job_input_data_stats(
        JobID(job_id),
        input_data_stats.json(),
    )


def save_input_data_stats_to_state_db(
    job_id: str,
    input_data_stats: InputDataStats,
) -> None:
    asyncio.run(
        _save_input_data_stats_to_state_db_async(
            job_id,
            input_data_stats,
        ))


async def _save_job_storage_path_to_state_db_async(
    job_id: str,
    storage_path: str,
) -> None:
    service = await _open_ml_job_service()
    await service.save_job_storage_path(JobID(job_id), storage_path)


def save_job_storage_path_to_state_db(
    job_id: str,
    storage_path: str,
) -> None:
    asyncio.run(_save_job_storage_path_to_state_db_async(
        job_id,
        storage_path,
    ))


def update_graph_config_from_graph_input(
    graph: GraphConfig,
    graph_input: GraphDataComputeInput,
) -> GraphConfig:
    """Timestamp unit could be updated during the graph refresh workflow.
    This function updates the graph config with the potentially updated
    timestamp unit from the graph input.

    Args:
        graph: The graph config to update.
        graph_input: The graph input data which contains updated timestamp
            unit from the graph refresh workflow.

    Returns:
        The updated graph config.
    """
    from rest.config.table_v2 import _decompose_timestamp_format
    updated_graph = copy.deepcopy(graph)
    # Update the graph config. v2 cols use Column.timestamp_format
    # (Optional[Union[TimestampUnit, str]]); only TimestampUnit enums map
    # to the legacy v1 `timestamp_unit` field on the GraphConfig table.
    for table_id, table in graph_input.tables.items():
        alias = graph_input.table_aliases[table_id]
        table_config = updated_graph[alias]
        all_cols = [
            *table_config.cols, *table_config.fkeys, table_config.pkey,
            table_config.time_col, table_config.end_time_col
        ]
        for col in table.cols:
            _, unit_str = _decompose_timestamp_format(col)
            for col_config in all_cols:
                if col_config and col_config.name == col.name:
                    if unit_str is not None:
                        logger.info(
                            "Update timestamp unit for column %s in table %s",
                            col.name, alias)
                        col_config.timestamp_unit = unit_str
                    col_config.multicategorical_delimiter = (
                        col.multicategorical_delimiter)

    return updated_graph
