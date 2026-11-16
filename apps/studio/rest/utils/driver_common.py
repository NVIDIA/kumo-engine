# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import os.path as osp
from contextlib import asynccontextmanager, contextmanager
from copy import copy, deepcopy
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)

import kumoapi.source_table as source_table_api
from kumo_core.types import PredictionArtifactType, TaskType
from kumo_core.types.schema import (
    TS_FORMAT_METADATA,
    BaseColumnConfig,
    ColumnArgs,
)
from kumoapi.source_table import SourceTableType, TableName
from kumoapi.typing import Stype

from async_utils.sync import blocking_func
from io_utils import logging
from io_utils.encryption import EncryptedStr
from io_utils.json import load_json_async
from kumo.config.graph import GraphConfig, TableMetadataConfig
from kumo.connector import Connector, ParquetConnector
from kumo.connector.connector import Dict
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from kumo.graph import Graph
from kumo.home import HomeDirManager
from monitoring.tracing import TraceOptions, trace
from rest.backends.data_backend import connector_activity_registry
from rest.backends.ml_backend.prediction_artifacts import (
    resolve_prediction_artifact,
)
from rest.config.batch_prediction_job import get_bp_columns_by_task_type
from rest.config.credentials import Credentials
from rest.config.data_backend import (
    ConnectorExecutionRef,
    ConnectorOpenSpec,
    ConnectorSpec,
    ResolvedConnectorSpec,
)
from rest.config.file_type import FileType
from rest.config.table_v2 import (
    ADLSSourceTable,
    BigQuerySourceTable,
    Column,
    CustomSQLSourceTable,
    DatabricksSourceTable,
    GCSSourceTable,
    S3SourceTable,
    SnowflakeSourceTable,
    TableDefinition,
    TimestampUnit,
)
from rest.utils.table import ColStatsSet
from statedb.config.common import ConnectorID
from statedb.config.table import TableDataVersion
from statedb.config.timestamp import ResourceTimestamps, now_utc

PREDICTIONS_SOURCE_TABLE_NAME = 'predictions'
TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME = 'transformed'

log = logging.getLogger(__name__)


@trace(TraceOptions(trace_params=False))
@connector_activity_registry.connector_resolution_helper
def get_connector_from_spec(connector_spec: ConnectorOpenSpec) -> Connector:
    if isinstance(connector_spec, ConnectorExecutionRef):
        from rest.backends.data_backend import connector_broker
        return get_connector_from_spec(
            connector_broker.connector_credential_broker.resolve(
                connector_spec))
    if isinstance(connector_spec, ResolvedConnectorSpec):
        connector = cast(Connector,
                         Connector.from_config(connector_spec.connector_config))
        connector.require_explicit_credentials()
        if connector_spec.credentials is not None:
            connector.set_credentials(connector_spec.credentials.as_dict())
        return connector
    connector = cast(Connector,
                     Connector.from_config(connector_spec.connector_config))
    return connector


@connector_activity_registry.connector_resolution_helper
async def get_connector_from_spec_async(
    connector_spec: ConnectorOpenSpec,
    *,
    resolved_factory: Callable[[ConnectorOpenSpec],
                               Connector] = get_connector_from_spec,
) -> Connector:
    """Resolve a connector without blocking an async activity event loop."""
    if isinstance(connector_spec, ConnectorExecutionRef):
        from rest.backends.data_backend import connector_broker
        resolved = await connector_broker.connector_credential_broker.resolve_async(
            connector_spec)
        return resolved_factory(resolved)
    return resolved_factory(connector_spec)


@blocking_func(DATA_BACKEND_THREAD_POOL)
def _open_connector(connector: Connector) -> Connector:
    """Open a connector without blocking an async activity event loop."""
    return connector.open()


@blocking_func(DATA_BACKEND_THREAD_POOL)
def _close_connector(connector: Connector) -> Connector:
    """Close a connector without blocking an async activity event loop."""
    return connector.close()


@asynccontextmanager
async def async_connector_lifecycle(
    connector: Connector,
    *,
    open_connector: bool = False,
) -> AsyncIterator[Connector]:
    """Close an activity-local connector on every exit path.

    Connector lifecycle methods are synchronous and may perform provider or
    filesystem I/O.  Run the complete public ``open``/``close`` boundary in
    the data-backend thread pool.  If work and cleanup both fail, preserve the
    work failure and record only exception types from the cleanup failure.
    """
    primary_error: BaseException | None = None
    try:
        if open_connector:
            await _open_connector.coroutine(connector)
        yield connector
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            await _close_connector.coroutine(connector)
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            log.warning(
                'Connector cleanup failed while preserving primary failure '
                '(primary_error_type=%s cleanup_error_type=%s)',
                type(primary_error).__name__,
                type(cleanup_error).__name__,
            )


@connector_activity_registry.connector_resolution_helper
def hydrate_graph_config_for_activity(
    graph_config: Any,
    references: Mapping[ConnectorID, ConnectorExecutionRef],
) -> tuple[Any, Dict[ConnectorID, Credentials]]:
    """Resolve a reference-only graph config once inside an activity.

    The returned graph config and credentials are process-local.  Keeping
    both products of one broker resolution together avoids a configuration /
    credential rotation race and prevents either from entering workflow
    history.
    """
    from rest.backends.data_backend import connector_broker

    hydrated = deepcopy(graph_config)
    connector_configs: dict[str, Any] = {}
    credentials: Dict[ConnectorID, Credentials] = {}
    if graph_config.data_source_dict:
        raise TypeError('Workflow graph configs must not contain connector '
                        'configurations')
    for connector_id_value, reference in references.items():
        connector_id = ConnectorID(connector_id_value)
        if not isinstance(reference, ConnectorExecutionRef):
            raise TypeError(
                'Activity graph configs must contain ConnectorExecutionRef '
                'values')
        if reference.connector_id != connector_id:
            raise ValueError(
                'Activity graph connector key must match reference '
                'connector_id')
        resolved = connector_broker.connector_credential_broker.resolve(
            reference)
        connector = get_connector_from_spec(resolved)
        primary_error: BaseException | None = None
        try:
            connector_configs[str(connector_id)] = connector.config()
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                connector.close()
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                log.warning(
                    'Connector cleanup failed while preserving primary '
                    'failure (primary_error_type=%s cleanup_error_type=%s)',
                    type(primary_error).__name__,
                    type(cleanup_error).__name__,
                )
        if (isinstance(resolved, ResolvedConnectorSpec)
                and resolved.credentials is not None):
            credentials[connector_id] = resolved.credentials

    hydrated.data_source_dict = connector_configs
    return hydrated, credentials


@contextmanager
def authorized_graph_for_activity(
    graph_config: Any,
    references: Mapping[ConnectorID, ConnectorExecutionRef],
) -> Iterator[Graph]:
    """Build and dispose an identity-authorized graph inside an activity.

    ``graph_config`` must be the reference-only form used in workflow
    payloads.  Connector configurations and credentials are resolved together
    through the activity-local broker and exist only for the lifetime of this
    context manager.  Every connector is closed, including when construction,
    provider I/O, cancellation, or a sibling connector cleanup fails.
    """
    if graph_config.data_source_dict:
        raise TypeError('Workflow graph configs must not contain connector '
                        'configurations')

    hydrated = deepcopy(graph_config)
    connectors: list[Connector] = []
    primary_error: BaseException | None = None
    try:
        connector_by_id: dict[str, Connector] = {}
        for connector_id_value, reference in references.items():
            connector_id = ConnectorID(connector_id_value)
            if not isinstance(reference, ConnectorExecutionRef):
                raise TypeError('Activity graph configs must contain '
                                'ConnectorExecutionRef values')
            if reference.connector_id != connector_id:
                raise ValueError(
                    'Activity graph connector key must match reference '
                    'connector_id')
            connector = get_connector_from_spec(reference)
            connectors.append(connector)
            connector_by_id[str(connector_id)] = connector

        table_connector_ids = {
            str(table.data_source)
            for table in graph_config.tables
        }
        if set(connector_by_id) != table_connector_ids:
            raise ValueError(
                'Activity graph connector references must exactly match its '
                'table data sources')
        hydrated.data_source_dict = connector_by_id
        yield Graph.from_config(hydrated)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error: BaseException | None = None
        for connector in reversed(connectors):
            try:
                connector.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                log.warning(
                    'Activity connector cleanup failed '
                    '(cleanup_error_type=%s)',
                    type(error).__name__,
                )
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error


def fail_closed_graph_from_config(graph_config: Any) -> Graph:
    """Reconstruct a request-local graph that cannot use ambient identity.

    Request processes may consume credential-free target metadata returned by
    a one-shot activity. They must never turn that metadata into a provider
    capability. Explicit-credential mode makes accidental lazy provider I/O
    fail before any ambient credential or legacy secret lookup.
    """
    graph = Graph.from_config(graph_config)
    for connector in graph.data_source_dict.values():
        connector.require_explicit_credentials()
    return graph


def credential_free_graph_config_for_activity(graph: Graph) -> GraphConfig:
    """Snapshot graph metadata without resolving lazy provider versions.

    ``Table.config()`` reads the public ``data_version`` property, which
    opens a connector when its cached value is absent. Request processes need
    to serialize the graph before handing it to an authorized activity, so
    use a shallow table copy with a temporary non-null marker and then restore
    the actual cached value on the resulting config. No provider method is
    called and no credentials are added.
    """
    table_configs = []
    for table in graph.tables:
        cached_data_version = getattr(table, '_data_version', None)
        table_copy = copy(table)
        if cached_data_version is None:
            table_copy._data_version = '__unresolved_data_version__'
        table_config = table_copy.config()
        table_config.data_version = cached_data_version
        table_configs.append(table_config)

    return GraphConfig(
        name=graph.name,
        data_source_dict={
            str(connector_id): connector.config()
            for connector_id, connector in graph.data_source_dict.items()
        },
        tables=table_configs,
    )


def hydrate_predictive_query_for_activity(
    predictive_query_config: Any,
    references: Mapping[ConnectorID, ConnectorExecutionRef],
) -> tuple[Any, Dict[ConnectorID, Credentials]]:
    """Hydrate a workflow-safe predictive query only in an activity."""
    hydrated = deepcopy(predictive_query_config)
    hydrated.graph, credentials = hydrate_graph_config_for_activity(
        hydrated.graph,
        references,
    )
    hydrated = hydrated.with_credentials(credentials={
        key: value.as_dict()
        for key, value in credentials.items()
    })
    return hydrated, credentials


def get_current_timestamps() -> ResourceTimestamps:
    now = now_utc()
    return ResourceTimestamps(now, now)


def update_timestamps(timestamps: ResourceTimestamps) -> ResourceTimestamps:
    updated_timestamps = copy(timestamps)
    updated_timestamps.updated_at = now_utc()
    return updated_timestamps


def open_connectors(graph: Graph, credentials_dict: Dict[ConnectorID,
                                                         Credentials]) -> None:
    data_sources = graph.data_source_dict
    for (connector_name, connector) in data_sources.items():
        connector_id = ConnectorID(connector_name)
        credentials: Dict[str, EncryptedStr] = {}
        if connector_id in credentials_dict:
            credentials = credentials_dict[connector_id].as_dict()
        connector.set_credentials(credentials)
        connector.open(credentials)


def extract_source_table_type(
    source_table: SourceTableType
) -> Tuple[Union[ADLSSourceTable, BigQuerySourceTable, DatabricksSourceTable,
                 GCSSourceTable, S3SourceTable, SnowflakeSourceTable,
                 CustomSQLSourceTable], TableName, ConnectorID]:
    """Returns the source table, table name and connector id for the given
        `source_table`. Every external source table has a connector ID."""
    if isinstance(source_table, source_table_api.SnowflakeSourceTable):
        source_table = SnowflakeSourceTable(**dataclasses.asdict(source_table))
        source_table_name = source_table.table
        connector_id = source_table.snowflake_connector_id
    elif isinstance(source_table, source_table_api.DatabricksSourceTable):
        source_table = DatabricksSourceTable(**dataclasses.asdict(source_table))
        source_table_name = source_table.table
        connector_id = source_table.databricks_connector_id
    elif isinstance(source_table, source_table_api.BigQuerySourceTable):
        source_table = BigQuerySourceTable(**dataclasses.asdict(source_table))
        source_table_name = source_table.table_name
        connector_id = source_table.bigquery_connector_id
    elif isinstance(source_table, source_table_api.S3SourceTable):
        source_table = S3SourceTable(**dataclasses.asdict(source_table))
        connector_id = source_table.connector_id
        source_table_name = source_table.table_name
    elif isinstance(source_table, source_table_api.GCSSourceTable):
        source_table = GCSSourceTable(**dataclasses.asdict(source_table))
        connector_id = source_table.connector_id
        source_table_name = source_table.table_name
    elif isinstance(source_table, source_table_api.ADLSSourceTable):
        source_table = ADLSSourceTable(**dataclasses.asdict(source_table))
        connector_id = source_table.connector_id
        source_table_name = source_table.table_name
    elif isinstance(source_table, source_table_api.CustomSQLSourceTable):
        source_table = CustomSQLSourceTable(**dataclasses.asdict(source_table))
        source_table_name = source_table.table
        connector_id = source_table.connector_id
    elif isinstance(source_table, source_table_api.LocalFileSourceTable):
        # Reached from the training driver.
        # (`pquery_api_driver.py:2209`, `:2357`).
        raise ValueError(
            "A local file custom training table is readable only by a backend "
            "running in the calling process, not by this server.")
    else:
        raise ValueError(f"Invalid source table type: {source_table}")

    return (source_table, source_table_name, ConnectorID(connector_id))


def prediction_connector_from_job_id(job_id: str) -> ConnectorID:
    # Internal prediction artifacts still need a valid immutable connector ID
    # because they flow through the same graph/table reference model as
    # project-owned connectors.  Derive it deterministically so retries and
    # related statistics workflows address the same internal connector.
    digest = hashlib.md5(
        job_id.encode(),
        usedforsecurity=False,
    ).hexdigest()
    return ConnectorID(f'connector_{digest}')


def _as_column_args(config: BaseColumnConfig,
                    is_pkey: bool = False) -> ColumnArgs:
    if is_pkey:
        stype: Stype = Stype.ID
    else:
        assert config.stype is not None
        stype: Stype = config.stype
    return ColumnArgs(
        name=config.name,
        stype=stype,
        dtype=config.dtype,
        metadata=config.metadata,
        timestamp_unit=config.timestamp_unit,
        multicategorical_delimiter=config.multicategorical_delimiter,
    )


async def generate_bp_stats_workflow_args(
    job_id: str,
    job_home_dir: str,
    task_type: TaskType,
    is_temporal: bool,
    entity_col_name: str,
    use_transformed_output: bool = False,
    data_version_override: Optional[str] = None,
) -> Tuple[ConnectorSpec, TableDefinition, TableDataVersion, ColStatsSet]:
    r"""
    Generate the arguments for the workflow that generates the stats for the
    batch prediction table.

    Args:
        job_id (str): The id of the batch prediction job.
        job_home_dir (str): The home directory of the batch prediction job.
        task_type (TaskType): The type of task to generate stats for.
        is_temporal (bool): Whether the task is temporal.
        entity_col_name (str): The user-facing entity column name. Raw
            predictions use their canonical physical ``ENTITY`` column;
            transformed predictions preserve this name.
        use_transformed_output (bool): Whether to use the transformed output.
        data_version_override (Optional[str]): Stable version to use when the
            raw artifact was written directly to a customer destination and
            therefore has no Kumo-internal Parquet path.

    Returns:
        Tuple[ConnectorSpec, TableDefinition, TableDataVersion, ColStatsSet]:
            The arguments for the workflow that generates the stats for the
            batch prediction table.
    """
    bp_connector_id = prediction_connector_from_job_id(job_id)

    home_dir_mngr = HomeDirManager(job_home_dir)
    connector_root = home_dir_mngr.prediction_output_dir
    selected_data_version: str | None = None
    if not use_transformed_output and data_version_override is None:
        predictions = await resolve_prediction_artifact.coroutine(
            home_dir_mngr,
            PredictionArtifactType.PREDICTIONS,
            include_selection_ref=True,
        )
        if predictions is not None:
            # Artifact roots preserve the historical `predictions.parquet`
            # dataset name. Moving only the connector root keeps the logical
            # table name and schema/fingerprint contract unchanged.
            connector_root = osp.dirname(predictions.root)
            if predictions.selection_ref is not None:
                selected_data_version = predictions.selection_ref.manifest_sha256
    bp_connector_spec = ConnectorSpec(
        connector_id=bp_connector_id,
        connector_config=ParquetConnector(
            name=bp_connector_id,
            root_dir=connector_root,
        ).config(),
    )
    source_table_name = (TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME
                         if use_transformed_output else
                         PREDICTIONS_SOURCE_TABLE_NAME)

    # Raw prediction artifacts use the canonical ENTITY column.
    if not use_transformed_output:
        entity_col_name = 'ENTITY'

    # Get the col configs by task type for the raw predictions or from s3
    # for the transformed predictions
    cols = get_bp_columns_by_task_type(task_type, is_temporal, entity_col_name)
    pkey: Optional[str] = entity_col_name
    if use_transformed_output:
        metadata = await load_json_async(
            home_dir_mngr.internal_transformed_predictions_metadata_file,
            TableMetadataConfig)
        cols = [
            _as_column_args(c, is_pkey=c.name == entity_col_name)
            for c in metadata.cols
        ]
        col_names = [c.name for c in metadata.cols]
        if entity_col_name not in col_names:
            pkey = None

    # The internal connector owns the prediction output directory. The table
    # stores only its connector-relative dataset name.
    bp_src_table = S3SourceTable(
        file_type=FileType.PARQUET,
        connector_id=bp_connector_id,
        source=source_table_api.S3RelativeSource(path=source_table_name),
    )
    bp_v2_cols = []
    for col in cols:
        ts_format: Optional[Any] = None
        if col.timestamp_unit:
            ts_format = TimestampUnit(col.timestamp_unit)
        elif TS_FORMAT_METADATA in col.metadata:
            ts_format = col.metadata[TS_FORMAT_METADATA]
        bp_v2_cols.append(
            Column(name=col.name, stype=col.stype, dtype=col.dtype,
                   timestamp_format=ts_format,
                   multicategorical_delimiter=(col.multicategorical_delimiter)))
    bp_table_def = TableDefinition(
        cols=tuple(bp_v2_cols),
        source_table=bp_src_table,
        pkey=pkey,
    )

    if data_version_override is not None:
        bp_data_version = TableDataVersion(data_version_override)
    elif selected_data_version is not None:
        bp_data_version = TableDataVersion(selected_data_version)
    else:
        connector: Connector = await get_connector_from_spec_async(
            bp_connector_spec,
            resolved_factory=get_connector_from_spec,
        )
        async with async_connector_lifecycle(connector):
            bp_data_version = TableDataVersion(
                await connector._get_data_version.coroutine(source_table_name))
    bp_col_stats_set = ColStatsSet.BP

    return (bp_connector_spec, bp_table_def, bp_data_version, bp_col_stats_set)
