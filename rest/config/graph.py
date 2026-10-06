# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import List, Optional, Union

import kumoapi.graph as api
from pydantic import field_validator

from io_utils import kumo_dataclass
from kumo.config.graph import GraphConfig
from kumo.graph import Graph
from statedb.config.common import (
    BatchPredictionWorkflowLink,
    ConnectorID,
    EntityValidationMessage,
    GraphID,
    PredictiveQueryLink,
    TableID,
    to_table_id,
)
from statedb.config.timestamp import ResourceTimestamps
from statedb.rest.graph import ColumnGroup


@kumo_dataclass
class GraphResponse:
    id: GraphID
    name: str
    table_ids: List[TableID]
    col_groups: List[ColumnGroup]
    num_tables: int
    num_links: int
    timestamps: ResourceTimestamps
    description: str = ''

    # Sum of row count over all tables. May be empty if stats are not ready.
    # At the time of writing, this is not implemented for "the list graphs"
    # endpoint.
    num_rows: Optional[int] = None


@kumo_dataclass
class GraphConfigResponse:
    config: GraphConfig


@kumo_dataclass
class EdgeHealthColumn:
    table_id: TableID
    connector_id: ConnectorID
    table_name: str
    column_name: str

    @field_validator('table_id', mode='before')
    @classmethod
    def _validate_table_id(cls, v: str) -> TableID:
        return to_table_id(v)


@kumo_dataclass(frozen=True)
class EdgeStat:
    a_in_b: Optional[float]
    b_in_a: Optional[float]

    def reverse(self) -> 'EdgeStat':
        return EdgeStat(
            a_in_b=self.b_in_a,
            b_in_a=self.a_in_b,
        )

    def both_are_valid(self) -> bool:
        return self.a_in_b is not None and self.b_in_a is not None

    def both_are_less_or_equal(self, threshold: float) -> bool:
        if not self.both_are_valid():
            return True
        assert self.a_in_b is not None
        assert self.b_in_a is not None
        return self.a_in_b <= threshold and self.b_in_a <= threshold

    def either_is_less_or_equal(self, threshold: float) -> bool:
        if not self.both_are_valid():
            return True
        assert self.a_in_b is not None
        assert self.b_in_a is not None
        return self.a_in_b <= threshold or self.b_in_a <= threshold


@kumo_dataclass
class EdgeHealth:
    # Col A is typically the primary key
    col_a: EdgeHealthColumn
    col_b: EdgeHealthColumn
    link_type: str
    matches: EdgeStat
    percents: EdgeStat
    edge_count: int

    def reverse(self) -> 'EdgeHealth':
        return EdgeHealth(
            col_a=self.col_b,
            col_b=self.col_a,
            link_type=self.link_type,
            matches=self.matches.reverse(),
            percents=self.percents.reverse(),
            edge_count=self.edge_count,
        )


@kumo_dataclass
class EdgeHealthStats:
    is_ready: bool = False
    link_stats: Optional[List[EdgeHealth]] = None


@kumo_dataclass(frozen=True)
class Edge(api.Edge):
    @property
    def fully_qualified_name(self) -> str:
        return f"{self.src_table}.{self.fkey}.{self.dst_table}"


def get_connector_ids_from_graph(
        graph: Union[GraphConfig, Graph]) -> List[ConnectorID]:
    connector_names = graph.data_source_dict.keys()
    return [ConnectorID(name) for name in connector_names]


@kumo_dataclass
class ValidateGraphResponse:
    """Describes the predictive queries that will be impacted
    by a given graph edit.
    """
    graph_error_message: Optional[str]

    # The following lists are empty if graph_error_message is provided
    modified_predictive_queries: List[PredictiveQueryLink] = field(
        default_factory=list)
    broken_predictive_queries: List[EntityValidationMessage] = field(
        default_factory=list)
    modified_batch_prediction_workflows: List[
        BatchPredictionWorkflowLink] = field(default_factory=list)
    broken_batch_prediction_workflows: List[EntityValidationMessage] = field(
        default_factory=list)
