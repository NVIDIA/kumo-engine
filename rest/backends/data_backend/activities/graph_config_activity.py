# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Dict

from kumo.config.graph import GraphConfig, TableConfig
from rest.backends.data_backend import connector_activity_registry
from rest.config.graph_v2 import GraphDataComputeInput
from rest.driver import graph_logic
from rest.utils.driver_common import authorized_graph_for_activity

_FORBIDDEN_MANAGEMENT_CONFIG_FIELD_PARTS = (
    'access_key',
    'api_key',
    'authorization',
    'client_id',
    'client_secret',
    'credential',
    'group_id',
    'identity',
    'password',
    'principal',
    'private_key',
    'secret',
    'service_account',
    'token',
    'username',
)


def assert_public_connector_target_config(
    value: Any,
    path: str = 'connector_config',
) -> None:
    """Fail closed if a one-shot management result contains identity data.

    Long-running data and ML workflows carry only ``ConnectorExecutionRef``.
    A synchronous one-shot management workflow may return already-public
    connector target metadata for request parsing, but never authentication,
    identity, principal, group, credential, secret, or token fields.
    """
    if dataclasses.is_dataclass(value):
        items = ((field.name, getattr(value, field.name))
                 for field in dataclasses.fields(value))
    elif isinstance(value, Mapping):
        items = value.items()
    else:
        items = None

    if items is not None:
        for key, item in items:
            normalized = str(key).casefold().replace('-', '_')
            if any(part in normalized
                   for part in _FORBIDDEN_MANAGEMENT_CONFIG_FIELD_PARTS):
                raise TypeError(
                    f'One-shot management connector config contains '
                    f'non-public field {path}.{key}')
            assert_public_connector_target_config(item, f'{path}.{key}')
        return

    if isinstance(value, Sequence) and not isinstance(value,
                                                      (str, bytes, bytearray)):
        for index, item in enumerate(value):
            assert_public_connector_target_config(item, f'{path}[{index}]')


def make_graph_config_from_compute_input(
    graph_input: GraphDataComputeInput,
    *,
    connector_configs: Dict[str, Any],
) -> GraphConfig:
    graph_def = graph_input.graph_definition
    if graph_def is None:
        raise ValueError('Graph definition is required')

    col_index = graph_logic._make_col_index_v2(graph_def.tables)
    links = graph_logic._make_fkeys_v2(
        col_index,
        graph_def.col_groups,
        graph_def.tables,
    )
    tables = []
    for table_name, definition in graph_def.tables.items():
        pkey = definition.primary_key_col()
        time_col = definition.time_column()
        end_time_col = definition.end_time_column()
        fkeys = links[table_name]
        tables.append(
            TableConfig(
                table_name=str(table_name),
                data_source=str(definition.get_connector_id()),
                pkey=pkey.config() if pkey else None,
                fkeys=[fkey.config() for fkey in fkeys],
                cols=[
                    col.config() for col in graph_logic._convert_cols_v2(
                        definition.cols, pkey, fkeys)
                ],
                time_col=time_col.config() if time_col else None,
                end_time_col=(end_time_col.config() if end_time_col else None),
                source_name=str(definition.get_source_table_name()),
                # Carry the Custom SQL marker so the graph rebuilt below
                # (``Graph.from_config`` then ``graph.config()``) resolves
                # ``data_version`` from the SQL identity instead of probing the
                # synthetic source name as a physical table (a 404 on BigQuery).
                is_sql=definition.is_sql,
            ))

    return GraphConfig(
        name=graph_input.graph_display_name,
        data_source_dict=connector_configs,
        tables=tables,
    )


@connector_activity_registry.connector_authorized_activity
def build_graph_config_activity(
    graph_input: GraphDataComputeInput, ) -> GraphConfig:
    """Build public target metadata for a synchronous management request.

    This is deliberately not a long-running job payload. The resulting job
    input is rebound to ``ConnectorExecutionRef`` and strips these native
    connector configs before it is submitted to a durable job workflow.
    """
    reference_only = make_graph_config_from_compute_input(
        graph_input,
        connector_configs={},
    )
    with authorized_graph_for_activity(reference_only,
                                       graph_input.connectors) as graph:
        # Serializing here pins every lazy table data version while the
        # connector has its project identity. The request receives only
        # credential-free target metadata and canonical versions.
        hydrated = graph.config()
    for connector_config in hydrated.data_source_dict.values():
        assert_public_connector_target_config(connector_config)
    return hydrated
