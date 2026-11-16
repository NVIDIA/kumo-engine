# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from typing import Dict, List, Optional, Tuple

from kumo_core.types import ConnectorType, EdgeStatType
from kumo_core.types.schema import (
    IngestedTableFingerprint,
    SqlTableFingerprint,
    TableFingerprint,
)

from async_utils.instrument import measure_blocking_code
from io_utils import logging
from kumo.connector.connector import Connector
from kumo.data import custom_sql, get_active_cache
from kumo.data.backend_cache import BackendCacheManager
from kumo.data.cache.graph_cache import (
    GraphEdgeStats,
    SingleEdgeStats,
    SourceTableColumn,
)
from rest.backends import CUSTOM_SQL_MAX_STALENESS_HOURS
from rest.backends.data_backend import connector_activity_registry
from rest.config.data_backend import ConnectorExecutionRef, ConnectorOpenSpec
from rest.config.graph import EdgeHealth, EdgeHealthColumn, EdgeStat
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.table_v2 import TableDefinition
from rest.utils.driver_common import (
    async_connector_lifecycle,
    get_connector_from_spec,
    get_connector_from_spec_async,
)
from statedb.config.common import TableID, TableName
from statedb.config.table import TableDataVersion

log = logging.getLogger(__name__)


def _connector_type_of(connector: Connector) -> ConnectorType:
    """Warehouse ``ConnectorType`` of a resolved connector instance.

    Derived from the concrete connector class rather than a ``config.type``
    field: a resolved spec's ``connector_config`` is the connector's NATIVE
    config (``ConfigMixin.config()``, reflected from ``__init__``), which
    carries no ``type`` field, so this mirrors the isinstance dispatch used
    elsewhere for resolved connectors (see
    ``connector_api_driver._connector_config_for_sdk``). Only warehouse
    connectors expose a Custom-SQL dialect, so anything else fails loud.
    """
    from kumo.connector import (
        BigQueryConnector,
        DatabricksConnector,
        SnowflakeConnector,
    )
    if isinstance(connector, SnowflakeConnector):
        return ConnectorType.SNOWFLAKE
    if isinstance(connector, DatabricksConnector):
        return ConnectorType.DATABRICKS
    if isinstance(connector, BigQueryConnector):
        return ConnectorType.BIGQUERY
    raise ValueError('Custom SQL Tables are not supported for '
                     f'{type(connector).__name__}')


@connector_activity_registry.connector_resolution_helper
def _connector_type_from_spec(
        connector_spec: ConnectorOpenSpec) -> ConnectorType:
    """Warehouse type of a connector spec, for picking the SQL dialect.

    A resolved spec (``ConnectorSpec`` / ``ResolvedConnectorSpec``) carries a
    ``connector_config`` that is the connector's NATIVE config, so the warehouse
    type is derived from the connector INSTANCE (via :func:`_connector_type_of`)
    rather than a nonexistent ``config.type`` field.

    A credential-free ``ConnectorExecutionRef`` carries no config, so it is
    first resolved from the durable connector row through the activity-only
    ``ConnectorCredentialBroker`` (no credential is returned to the request
    process — the resolved spec is only used to build the connector instance and
    read its class here). This branch may run only inside a Temporal activity;
    async activities should use :func:`resolve_connector_and_type_async` to
    avoid blocking the event loop.
    """
    if isinstance(connector_spec, ConnectorExecutionRef):
        from rest.backends.data_backend import connector_broker
        connector_spec = connector_broker.connector_credential_broker.resolve(
            connector_spec)
    return _connector_type_of(get_connector_from_spec(connector_spec))


@connector_activity_registry.connector_resolution_helper
async def _connector_type_from_spec_async(
        connector_spec: ConnectorOpenSpec) -> ConnectorType:
    """Async sibling of :func:`_connector_type_from_spec`.

    Resolves a ``ConnectorExecutionRef`` through the broker without blocking the
    async activity event loop; a config-bearing spec derives its warehouse type
    from the connector instance with no broker round-trip.
    """
    if isinstance(connector_spec, ConnectorExecutionRef):
        from rest.backends.data_backend import connector_broker
        connector_spec = await (connector_broker.connector_credential_broker.
                                resolve_async(connector_spec))
    return _connector_type_from_spec(connector_spec)


@connector_activity_registry.connector_resolution_helper
def resolve_connector_and_type(
        connector_spec: ConnectorOpenSpec) -> Tuple[Connector, ConnectorType]:
    """Resolve a connected connector and its warehouse type from a spec/ref.

    Inside an activity a ``ConnectorExecutionRef`` is resolved once through the
    activity-only credential broker; a config-bearing spec is used directly.
    Both the connector object and its Custom-SQL dialect come from that single
    resolution, so credentials are never resolved more than once. Only
    warehouse connectors expose a Custom-SQL dialect, so this is used on the
    SQL authoring path (validate + query discovery/sample) alone.
    """
    if isinstance(connector_spec, ConnectorExecutionRef):
        from rest.backends.data_backend import connector_broker
        connector_spec = connector_broker.connector_credential_broker.resolve(
            connector_spec)
    connector = get_connector_from_spec(connector_spec)
    return connector, _connector_type_of(connector)


@connector_activity_registry.connector_resolution_helper
async def resolve_connector_and_type_async(
        connector_spec: ConnectorOpenSpec) -> Tuple[Connector, ConnectorType]:
    """Async sibling of :func:`resolve_connector_and_type`.

    Resolves the ``ConnectorExecutionRef`` through the broker without blocking
    the async activity event loop, then derives the connector and warehouse
    type from that single resolution.
    """
    if isinstance(connector_spec, ConnectorExecutionRef):
        from rest.backends.data_backend import connector_broker
        connector_spec = await (connector_broker.connector_credential_broker.
                                resolve_async(connector_spec))
    connector = get_connector_from_spec(connector_spec)
    return connector, _connector_type_of(connector)


def get_data_operation_description() -> str:
    return 'Local data operation'


def compose_table_ingest_wf_id(
    table_fingerprint: TableFingerprint,
    data_version: TableDataVersion,
) -> str:
    # Both physical (`IngestedTableFingerprint`) and Custom SQL
    # (`SqlTableFingerprint`) tables expose ``ingest_path_fpid`` and share the
    # ingest workflow; a SQL table's fpid lives in a disjoint ``sql`` namespace.
    assert isinstance(table_fingerprint,
                      (IngestedTableFingerprint, SqlTableFingerprint))
    return 'table-ingest-' + ".".join([
        table_fingerprint.ingest_path_fpid,
        data_version,
    ])


def data_backend_cache() -> BackendCacheManager:
    """Just a convenient helper for using the cache with proper type."""
    return get_active_cache()


async def _get_sql_table_data_version(
        connector_spec: ConnectorOpenSpec, table_def: TableDefinition,
        refresh_table_version: bool) -> TableDataVersion:
    """Change-detection version for a Custom SQL Table.

    ``data_version = md5(normalized SQL + sorted referenced-table versions +
    output cols/overrides)``. The referenced external tables come from the
    sqlglot parse of the current SQL, and each one's version is resolved
    recursively through :func:`get_table_data_version` (physical path — a SQL
    table cannot reference another SQL table). A referenced table whose version
    can't be resolved, and a query sqlglot can't parse at all, are each folded
    in as a fresh unique sentinel, so uncertainty forces a conservative
    re-materialization rather than risking a stale cache hit (AC6.5).

    ``referenced_tables_or_none`` renders each identifier *dialect-quoted and
    fully qualified* exactly as written (BigQuery backticks a hyphenated project
    id; Snowflake keeps a case-preserving double-quote). The connector version
    probe, however, looks a table up by its BARE name inside the connector's own
    configured project.dataset / db.schema — BigQuery re-qualifies via
    ``full_table_name``, Snowflake pins ``TABLE_SCHEMA`` to the connector schema,
    Databricks re-qualifies via ``_fully_qualify_table_name`` — so a quoted or
    qualified string never matches. Each ref is therefore reduced to its
    dialect-cased bare leaf (:func:`custom_sql.normalized_sql_identifier_leaf`,
    which strips quoting, preserves BigQuery's case-sensitive table names, and
    upper-/lower-folds for Snowflake/Databricks) before the probe. Without this,
    BigQuery's ``get_table`` rejected the quoted, over-qualified name, every
    referenced-table lookup threw, and the ``indeterminate`` sentinel below fired
    on *every* call — so the same graph/query recomputed a different data_version
    each time and batch prediction asked for a snapshot training never
    materialized.

    Referenced-table versions come from the connector's ``get_data_version``,
    exactly as for a physical Kumo table built on the same source: a reliable
    ``LAST_ALTERED`` for base tables, and the connector's hour-bucketed fallback
    for warehouse views. SQL tables therefore inherit raw-table change-detection
    granularity for view references (PRD §7) rather than a different one.

    On top of that precise detection, an aligned max-staleness bucket is always
    folded into the fingerprint's dedicated staleness field so ``data_version``
    rolls over — and re-materialization fires — at least every
    ``CUSTOM_SQL_MAX_STALENESS_HOURS`` regardless of detected change (see the
    inline comment below for why).
    """
    src = table_def.source_table
    # Async-safe: a ConnectorExecutionRef (the graph refresh/version path) is
    # resolved through the broker off the event loop rather than via the sync
    # `_connector_type_from_spec` broker call.
    connector_type = await _connector_type_from_spec_async(connector_spec)

    referenced_versions: Dict[str, str] = {}
    referenced = custom_sql.referenced_tables_or_none(src.sql, connector_type)
    if referenced is None:
        # SQL that no longer parses: the dependency set is unknown, so treat it
        # as changed via a fresh sentinel rather than hashing SQL+cols alone
        # (which would omit real inputs and could serve stale data).
        referenced_versions[
            '__unparsed__'] = f"indeterminate:{uuid.uuid4().hex}"
        log.warning(
            "Custom SQL table %s: SQL could not be parsed for dependency "
            "extraction; re-materializing conservatively.",
            table_def.get_source_table_name())
        referenced = []
    for ref in referenced:
        # Reduce the dialect-quoted, fully-qualified parse output to the
        # bare, dialect-cased leaf the connector probe actually resolves (see
        # the docstring): keeps BigQuery deterministic instead of throwing on
        # every quoted ref and collapsing to a fresh per-call sentinel.
        lookup = custom_sql.normalized_sql_identifier_leaf(ref, connector_type)
        try:
            referenced_versions[lookup] = str(await get_table_data_version(
                connector_spec, lookup,
                refresh_table_version=refresh_table_version))
        except Exception as exc:
            # Any failure to resolve a referenced table's version (missing
            # table, connector error) is treated as "changed": a fresh sentinel
            # guarantees a distinct data_version so the skip-guard
            # re-materializes instead of serving stale data.
            referenced_versions[lookup] = f"indeterminate:{uuid.uuid4().hex}"
            log.warning(
                "Custom SQL table %s: could not resolve version for "
                "referenced table %s (%r); re-materializing conservatively.",
                table_def.get_source_table_name(), lookup, exc)

    # Max-staleness floor: an aligned wall-clock window index, folded into the
    # fingerprint's dedicated staleness field (NOT the referenced-table map), so
    # re-materialization fires at least every CUSTOM_SQL_MAX_STALENESS_HOURS
    # regardless of detected change. It layers an N-hour ceiling on top of precise
    # change-detection without weakening the fresh-uuid sentinels above
    # (indeterminate cases still re-ingest on every call).
    #
    # The floor is unconditional because the parsed referenced-table set is an
    # incomplete/unreliable proxy for what the query actually reads, in two ways
    # we cannot recover from parsing — so their versions can stay static forever
    # while the query's real data changes underneath:
    #   1. SQL functions/constructs that pull data with no recoverable table
    #      reference (TVFs, ``read_files()``, ``generate_series``, no ``FROM``).
    #      These can appear ALONGSIDE ordinary parseable tables — i.e. a
    #      NON-EMPTY but incomplete set — so the extracted refs' versions may
    #      never move even as the un-extractable source keeps changing (the
    #      no-``FROM`` / empty-set case is just the degenerate extreme).
    #   2. Warehouse views: the view's underlying data can change without the
    #      parsed reference's version moving in a way our change-detection sees.
    # Because neither is recoverable from the parse, a blanket time floor is the
    # only safe bound. The bucket is coarse and aligned, so within one window it
    # is constant → stable inputs still yield a stable data_version (cache hit,
    # no churn); only crossing a window boundary rolls it over and triggers
    # exactly one re-ingest. ``CUSTOM_SQL_MAX_STALENESS_HOURS`` is clamped to >= 1
    # at config load, so the divisor is always positive.
    staleness_bucket = int(
        time.time()) // (CUSTOM_SQL_MAX_STALENESS_HOURS * 3600)

    # Match the cols dict the export projects/casts by (``table_fp.cols``), so a
    # dtype-override edit changes the version even though the path-id does not.
    cols = {c.name: c.dtype for c in table_def.cols}
    return TableDataVersion(
        custom_sql.custom_sql_data_version(
            src.sql, connector_type, referenced_versions, cols,
            max_staleness_bucket=staleness_bucket))


@measure_blocking_code
async def get_table_data_version(
        connector_spec: ConnectorOpenSpec, source_table_name: str,
        refresh_table_version: bool = False,
        table_def: Optional[TableDefinition] = None) -> TableDataVersion:
    # Custom SQL tables have no physical source table: their version is a
    # content hash over the SQL + referenced-table versions + output cols, not a
    # connector probe of the synthetic name. ``table_def`` is threaded in by the
    # refresh/version callers (which hold it); the recursive referenced-table
    # resolution above passes no ``table_def`` and takes the physical path.
    if table_def is not None and table_def.is_sql:
        return await _get_sql_table_data_version(connector_spec, table_def,
                                                 refresh_table_version)
    # Don't open connector if we don't need to, i.e. because the
    # caller did not request a refresh for a new version
    if not refresh_table_version:
        version = (await get_active_cache().table_version_cache.get.coroutine(
            connector_spec.connector_id, source_table_name))
        if version is not None:
            return TableDataVersion(version)

    # if we still don't have a version since there wasn't one
    # cached, or the caller wants a new version, get the version
    # from the connector
    connector = await get_connector_from_spec_async(
        connector_spec, resolved_factory=get_connector_from_spec)
    # Retrieve fresh table data version from connector if
    # refresh_table_version is True. However, do not save the updated
    # data version to backend cache yet. This is only saved after
    # ingestion workflow completes, so the version cached represents
    # the latest ingested version.
    async with async_connector_lifecycle(connector):
        version = await connector.get_data_version.coroutine(
            source_table_name, refresh_table_version, False)
    return TableDataVersion(version)


def parse_edge_stats_result(graph_input: GraphDataComputeInput,
                            edge_stats: GraphEdgeStats) -> List[EdgeHealth]:
    """Parse edge stats fetch from the backend cache for the given graph.

    Iterates ``graph_input.graph_definition.col_groups`` (name-keyed
    ``ColumnKeyGroup``) and looks up table schemas via
    ``graph_definition.tables`` (also name-keyed).
    """
    edge_health_results = []
    tables_by_name = graph_input.graph_definition.tables
    # Inverse of ``graph_input.table_aliases`` (TableID -> TableName) so
    # we can populate ``EdgeHealthColumn.table_id`` from the name-keyed
    # ColumnKey we now iterate over.
    name_to_table_id: Dict[TableName, TableID] = {
        name: tid
        for tid, name in graph_input.table_aliases.items()
    }
    for col_group in graph_input.graph_definition.col_groups:
        if len(col_group.columns) == 0:
            continue

        # Find a pkey in the col group and use this as col1. Otherwise,
        # use the first column as col1
        col_1_idx: int = 0
        for i, col in enumerate(col_group.columns):
            table_config = tables_by_name[col.table_name]
            if col.col_name == table_config.pkey:
                col_1_idx = i
                break
        col1 = col_group.columns[col_1_idx]
        table1 = tables_by_name[col1.table_name]
        src_col = SourceTableColumn(table1.get_connector_id(),
                                    table1.get_source_table_name(),
                                    col1.col_name)

        # Get edge stats between col1 and every other column
        for i, col2 in enumerate(col_group.columns):
            if col_1_idx == i:
                continue

            table2 = tables_by_name[col2.table_name]
            dst_col = SourceTableColumn(table2.get_connector_id(),
                                        table2.get_source_table_name(),
                                        col2.col_name)

            stats = edge_stats.get_stats(src_col, dst_col)

            # Check the reverse relationship if missing
            reverse: bool = False
            if stats is None:
                stats = edge_stats.get_stats(dst_col, src_col)
                reverse = True

                # Log error if stats are missing to alert oncall engineer
                if stats is None:  # type: ignore
                    log.error(
                        "Failed to fetch stats for edge between %s and %s",
                        col1, col2)
                    continue

            edge_health_column_1 = EdgeHealthColumn(
                table_id=name_to_table_id[col1.table_name],
                connector_id=table1.get_connector_id(),
                table_name=col1.table_name,
                column_name=col1.col_name,
            )
            edge_health_column_2 = EdgeHealthColumn(
                table_id=name_to_table_id[col2.table_name],
                connector_id=table2.get_connector_id(),
                table_name=col2.table_name,
                column_name=col2.col_name,
            )

            edge_health = EdgeHealth(
                col_a=(edge_health_column_1
                       if not reverse else edge_health_column_2),
                col_b=(edge_health_column_2
                       if not reverse else edge_health_column_1),
                link_type="Exact Match",
                matches=EdgeStat(
                    a_in_b=get_stat_value(stats,
                                          EdgeStatType.NUM_SRC_NONZERO_DEGREE),
                    b_in_a=get_stat_value(stats,
                                          EdgeStatType.NUM_DST_NONZERO_DEGREE),
                ),
                percents=EdgeStat(
                    a_in_b=get_stat_value(stats,
                                          EdgeStatType.FRAC_SRC_NONZERO_DEGREE,
                                          percent=True),
                    b_in_a=get_stat_value(stats,
                                          EdgeStatType.FRAC_DST_NONZERO_DEGREE,
                                          percent=True)),
                edge_count=int(
                    get_stat_value(stats, EdgeStatType.EDGE_COUNT) or 0),
            )
            if reverse:
                edge_health = edge_health.reverse()

            edge_health_results.append(edge_health)
    return edge_health_results


def get_stat_value(
    stats: SingleEdgeStats,
    stat_type: EdgeStatType,
    percent: bool = False,
) -> Optional[float]:
    if stat_type in stats.stats:
        try:
            return float(stats.stats[stat_type].value) * (100 if percent else 1)
        except TypeError as e:
            # Sometimes the stat is not a number. Root cause is not known.
            log.exception(e)
            return None
    return None
