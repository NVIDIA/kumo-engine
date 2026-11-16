# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple

import kumoapi.graph as graph_api
from kumo_core.types.schema import BaseColumnConfig
from kumoapi.table import TimestampUnit

from kumo.config.graph import GraphConfig, TableConfig
from rest.config.connector import ConnectorResourceConfigDB
from rest.config.graph_v2 import ColumnKey, ColumnKeyGroup, GraphDefinition
from rest.config.table_v2 import Column, TableDefinition
from statedb.config.common import ConnectorID, TableName


def _column_config_to_v2_column(config: BaseColumnConfig) -> Column:
    r"""Build a v2 ``Column`` from a kumo SDK ``BaseColumnConfig``.

    Skips the v1 ``ColumnArgs`` intermediate so this helper stays free of
    v1 types.
    """
    assert config.stype
    assert config.dtype
    ts_format: Optional[TimestampUnit] = None
    if config.timestamp_unit:
        ts_format = TimestampUnit(config.timestamp_unit)
    return Column(
        name=config.name, stype=config.stype, dtype=config.dtype,
        timestamp_format=ts_format,
        multicategorical_delimiter=(config.multicategorical_delimiter))


def _table_config_to_table_definition(
        table: TableConfig, connector_id: ConnectorID,
        connector_config: ConnectorResourceConfigDB) -> TableDefinition:
    r"""Build a v2 ``TableDefinition`` natively from a kumo SDK
    ``TableConfig`` and its connector create shape.

    Calls ``TableDefinition._build_from_inputs`` directly so this path
    never materializes a v1 table-args object.
    """
    assert table.table_name is not None
    assert table.source_name

    cols: List[Column] = [_column_config_to_v2_column(c) for c in table.cols]
    if table.pkey is not None:
        cols.append(_column_config_to_v2_column(table.pkey))
    for fkey in table.fkeys:
        cols.append(_column_config_to_v2_column(fkey))

    table_def = TableDefinition._build_from_inputs(
        connector_id=connector_id,
        connector_config=connector_config,
        source_table_name=table.source_name,
        cols=cols,
        pkey=table.pkey.name if table.pkey else None,
        time_col=table.time_col.name if table.time_col else None,
        end_time_col=None,
    )
    if table_def is None:
        raise ValueError(
            f"Failed to retrieve table {table.table_name} - source table "
            f"could not be found in connector "
            f"{connector_id}")
    return table_def


def export_kumo_graph_to_fixture(
    graph_config: GraphConfig,
    connectors: Dict[str, Tuple[ConnectorID, ConnectorResourceConfigDB]],
) -> GraphDefinition:
    r"""Convert a kumo SDK ``GraphConfig`` to a v2 ``GraphDefinition``.

    ``connectors`` maps ``ConnectorID`` to a connector create shape and is
    required so each table can be coerced through
    ``TableDefinition._build_from_inputs``. Callers
    (``build_graph_data_compute_input``, ``config2fixture``) already have
    this map.

    Returns a ``GraphDefinition`` whose:
    - ``tables`` is keyed by per-graph ``TableName`` (the SDK alias).
    - ``col_groups`` are name-keyed ``ColumnKeyGroup``\s derived from
      pkey/fkey relationships in ``graph_config``.
    """
    graph_name = graph_config.name
    assert graph_name

    # Build name-keyed TableDefinition map. SDK ``TableConfig.table_name``
    # is the per-graph alias and uniquely identifies the table within the
    # graph.
    tables_by_name: Dict[TableName, TableDefinition] = {}
    pkey_by_name: Dict[TableName, str] = {}
    for table in graph_config.tables:
        assert table.table_name is not None
        assert table.data_source is not None
        connector_id, connector_config = connectors[table.data_source]
        table_def = _table_config_to_table_definition(table, connector_id,
                                                      connector_config)
        tables_by_name[TableName(table.table_name)] = table_def
        if table.pkey is not None:
            pkey_by_name[TableName(table.table_name)] = table.pkey.name

    # Derive ColumnKeyGroup from pkey/fkey relationships. For each pkey
    # column, walk all other tables looking for fkeys that target it; emit
    # a single ColumnKeyGroup containing the pkey + matching fkey columns.
    relationships = []
    for pkey_table_name, pkey_col_name in pkey_by_name.items():
        # Walk the original SDK tables (not the v2 dict) so we can read
        # the original ``fkey.dst_table`` / ``fkey.dst_key`` fields.
        for other_table in graph_config.tables:
            if not other_table.fkeys:
                continue
            assert other_table.table_name is not None
            other_name = TableName(other_table.table_name)
            for fkey in other_table.fkeys:
                # ``fkey.dst_table`` is a v1-style fully-qualified name
                # (``{connector}.{table_name}``). Extract just the table
                # name and compare against the pkey table's name.
                assert fkey.dst_table
                _, dst_table_name = fkey.dst_table.rsplit('.', 1)
                if (dst_table_name == pkey_table_name
                        and fkey.dst_key == pkey_col_name):
                    relationships.append((
                        ColumnKey(table_name=other_name, col_name=fkey.name),
                        ColumnKey(table_name=pkey_table_name,
                                  col_name=pkey_col_name),
                    ))
    col_groups = [
        ColumnKeyGroup(columns=group.columns)
        for group in graph_api.group_column_relationships(relationships)
    ]

    return GraphDefinition(tables=tables_by_name, col_groups=col_groups)
