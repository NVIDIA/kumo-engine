# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Mapping, Optional, Union

from io_utils import kumo_dataclass
from kumo.config.predictive_query import PredictiveQueryV2Config
from kumo.data.cache.graph_cache import GraphVersion
from kumo.validation import ValidationResponse
from rest.config.credentials import Credentials
from statedb.config.common import ConnectorID, ProjectID, TableName
from statedb.config.table import TableDataVersion


@kumo_dataclass
class ConnectorSpec:
    """In-process configuration for a resolved internal connector."""

    __temporal_serializable__: ClassVar[bool] = False
    connector_id: ConnectorID
    connector_config: Any  # ConnectorConfig, skip validation
    is_project_upload: bool = False


@kumo_dataclass(frozen=True)
class ConnectorExecutionRef:
    """Opaque workflow capability for one project-owned connector.

    The random token is intentionally excluded from representations. StateDB
    persists only its hash; the raw value exists solely in trusted Temporal
    payloads and activity memory.
    """
    connector_id: ConnectorID
    project_id: ProjectID
    authorization_token: str = field(default='', repr=False)


@kumo_dataclass(frozen=True)
class ActivityAuthorizationRef:
    """Opaque authorization for a credential-bearing non-connector action."""

    authorization_token: str = field(repr=False)


@kumo_dataclass
class ValidatedPredictiveQuerySnapshot:
    """Credential-free result of activity-side PQuery validation.

    ``config.graph.data_source_dict`` is always empty. Table metadata, data
    versions, and the validated PQL representation are retained so request
    code can reconstruct the query without resolving a connector or touching
    a provider.
    """

    config: PredictiveQueryV2Config
    validation_response: ValidationResponse


def bind_graph_config_to_execution_refs(
    graph_config: Any,
    references: Mapping[ConnectorID, ConnectorExecutionRef],
) -> Any:
    """Return a workflow-safe graph config containing only connector refs.

    Graph and predictive-query configs still carry table/column metadata used
    by deterministic workflow orchestration.  Their ``data_source_dict`` must
    not duplicate a resolved connector configuration, however: the durable
    connector representation is exclusively ``ConnectorExecutionRef``.
    """
    by_id = {
        ConnectorID(connector_id): reference
        for connector_id, reference in references.items()
    }
    for connector_id, reference in by_id.items():
        if not isinstance(reference, ConnectorExecutionRef):
            raise TypeError(
                'Workflow graph connectors must be ConnectorExecutionRef '
                'values')
        if reference.connector_id != connector_id:
            raise ValueError(
                'Workflow graph connector key must match reference '
                'connector_id')

    configured_ids = {
        ConnectorID(connector_id)
        for connector_id in graph_config.data_source_dict
    }
    table_connector_ids = {
        ConnectorID(table.data_source)
        for table in graph_config.tables
    }
    if configured_ids and table_connector_ids != configured_ids:
        raise ValueError(
            'Workflow graph tables and data sources must reference the same '
            'connectors')
    if not configured_ids:
        configured_ids = table_connector_ids
    if configured_ids != set(by_id):
        raise ValueError(
            'Workflow graph connector references must exactly match its '
            'data sources')

    result = copy.deepcopy(graph_config)
    # References live in the explicitly typed GraphDataComputeInput rather
    # than this ``Dict[str, Any]`` field. Leaving it empty also prevents the
    # generic GraphConfig decoder from degrading nested refs into plain dicts.
    result.data_source_dict = {}
    return result


def bind_predictive_query_to_execution_refs(
    predictive_query_config: Any,
    references: Mapping[ConnectorID, ConnectorExecutionRef],
) -> Any:
    """Return a predictive-query config safe for job/workflow persistence."""
    result = copy.deepcopy(predictive_query_config)
    result.graph = bind_graph_config_to_execution_refs(
        result.graph,
        references,
    )
    return result


@dataclass(frozen=True, repr=False)
class ResolvedConnectorSpec:
    """In-process connector configuration with decrypted credentials.

    This type deliberately is not a Kumo/Pydantic dataclass so it cannot be
    accepted by the normal Temporal data converter as a workflow payload.
    """
    __temporal_serializable__: ClassVar[bool] = False
    connector_id: ConnectorID
    connector_config: Any
    credentials: Optional[Credentials]


ConnectorOpenSpec = Union[ConnectorExecutionRef, ConnectorSpec,
                          ResolvedConnectorSpec]


@kumo_dataclass
class GraphDataVersionInfo:
    graph_data_version: GraphVersion
    table_data_versions: Dict[TableName, TableDataVersion]
    # Name/alias of the graph at the time the snapshot was taken.
    # None if the graph had no alias set.
    graph_name: Optional[str] = None
