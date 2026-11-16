# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic.dataclasses import dataclass

from kumoapi.table import TableDefinition
from kumoapi.typing import _FORBID_EXTRA_CONFIG

TableName = str


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ColumnKey:
    """Reference to a column within a PQuery/Graph definition."""
    table_name: TableName
    col_name: str


@dataclass(frozen=True, config=_FORBID_EXTRA_CONFIG)
class ColumnKeyGroup:
    """Group of column keys to be linked together in a graph."""
    columns: Tuple[ColumnKey,
                   ...]  # Expected sorted, immutable; duplicates allowed


def group_column_relationships(
    relationships: Iterable[Tuple[ColumnKey, ColumnKey]],
) -> List[ColumnKeyGroup]:
    """Group source keys by their destination key.

    Group order follows the first occurrence of each destination. Columns are
    sorted for deterministic serialization, but duplicates remain visible to
    the graph validator.
    """
    grouped: Dict[ColumnKey, List[ColumnKey]] = {}
    for source, destination in relationships:
        grouped.setdefault(destination, [destination]).append(source)
    return [
        ColumnKeyGroup(
            tuple(
                sorted(columns, key=lambda key:
                       (key.table_name, key.col_name))))
        for columns in grouped.values()
    ]


@dataclass(frozen=True, eq=True, config=_FORBID_EXTRA_CONFIG)
class Edge:
    r"""A representation of an edge between tables in Kumo."""
    src_table: str
    fkey: str
    dst_table: str


class GraphSpecError(ValueError):
    """A graph definition violates a structural invariant."""
    def __init__(
            self,
            code: str,
            message: str,
            *,
            column: Optional[ColumnKey] = None,
            columns: Tuple[ColumnKey, ...] = (),
            edge: Optional[Edge] = None,
    ):
        self.code = code
        self.column = column
        self.columns = columns
        self.edge = edge
        super().__init__(message)


def validate_edge_endpoints(
    tables: Mapping[TableName, object],
    edges: Iterable[Edge],
) -> None:
    """Require every edge endpoint to name a table in the graph.

    Callers may build resolved :class:`Edge` objects directly or derive them
    from column groups. Validate the shared referential invariant before
    either representation is projected or submitted to a backend.
    """
    table_names = set(tables)
    available = sorted(table_names)
    for edge in edges:
        for role, table_name in (("source", edge.src_table), ("destination",
                                                              edge.dst_table)):
            if table_name not in table_names:
                raise GraphSpecError(
                    f"missing_{role}_table",
                    f"Edge {edge.src_table!r}.{edge.fkey!r} -> "
                    f"{edge.dst_table!r} references missing {role} table "
                    f"{table_name!r}; available tables: {available!r}.",
                    edge=edge,
                )


def resolve_foreign_keys(
    tables: Mapping[TableName, TableDefinition],
    column_groups: Sequence[ColumnKeyGroup],
) -> Tuple[Edge, ...]:
    """Resolve each column group into directed foreign-key edges.

    Each group must contain exactly one declared primary key. Every other
    member points to that key, including a foreign key declared on the same
    table. Edge order follows group and member order.
    """
    all_columns = {
        ColumnKey(str(table_name), column.name)
        for table_name, table in tables.items()
        for column in table.cols
    }
    primary_keys = {
        ColumnKey(str(table_name), table.pkey)
        for table_name, table in tables.items() if table.pkey is not None
    }
    edges: List[Edge] = []

    for column_group in column_groups:
        group = tuple(
            ColumnKey(str(column.table_name), str(column.col_name))
            for column in column_group.columns)
        keys = [column for column in group if column in primary_keys]
        if not keys:
            rendered = ", ".join(f"{column.table_name}.{column.col_name}"
                                 for column in group)
            raise GraphSpecError(
                "no_primary_key",
                f"column group has no primary key: {rendered}",
                columns=group,
            )
        if len(keys) > 1:
            raise GraphSpecError(
                "multiple_primary_keys",
                f"column group has multiple primary keys: {keys[0]}, {keys[1]}",
                columns=tuple(keys),
            )

        destination = keys[0]
        for source in group:
            if source not in all_columns:
                raise GraphSpecError(
                    "missing_column",
                    f"column {source} is not present in table {source.table_name!r}",
                    column=source,
                )
            if source == destination:
                continue
            edges.append(
                Edge(
                    src_table=source.table_name,
                    fkey=source.col_name,
                    dst_table=destination.table_name,
                ))

    return tuple(edges)


@dataclass(frozen=True)
class EdgeMatches:
    r"""
    Stats (either absolute values or percentages) about an edge relating to
    how many rows in the source have edges to the destination and vice versa.
    """
    src_in_dst: float
    dst_in_src: float


@dataclass(frozen=True)
class EdgeHealthStatistics:
    r"""Information about the health of an edge."""
    absolute_match_stats: EdgeMatches
    percent_match_stats: EdgeMatches
    total_num_edges: int


@dataclass
class EdgeHealthResponse:
    """Statistics about the health of each edge in a graph"""
    is_ready: bool
    statistics: Dict[str, EdgeHealthStatistics]


@dataclass(config=_FORBID_EXTRA_CONFIG)
class GraphDefinition:
    tables: Dict[TableName, TableDefinition]
    col_groups: List[ColumnKeyGroup]


def graph_definition_from_edges(
    tables: Mapping[TableName, TableDefinition],
    edges: Sequence[Edge],
    *,
    definition_type: type[GraphDefinition] = GraphDefinition,
) -> GraphDefinition:
    validate_edge_endpoints(tables, edges)
    relationships = []
    for edge in edges:
        dst_pkey = tables[edge.dst_table].pkey
        if dst_pkey is None:
            raise ValueError(f"The destination table {edge.dst_table} of edge "
                             f"{edge} does not have a primary key.")
        relationships.append((
            ColumnKey(edge.src_table, edge.fkey),
            ColumnKey(edge.dst_table, dst_pkey),
        ))
    return definition_type(
        tables=dict(tables),
        col_groups=group_column_relationships(relationships))


@dataclass
class GraphResource:
    # Graph resource identifier in the format of "graph-<16digit hex id>",
    # this is generated on the backend.
    id: str

    # Immutable metadata definition for a graph.
    graph: GraphDefinition

    # Optional human-friendly name alias for the graph. Graphs can be created
    # with or without name alias. However, once a graph resource sets the name
    # alias, it cannot be renamed. Name aliases must be unique, in other words,
    # 1:1 mapping is required between graph name aliases to GraphResource.id
    # Effectively, name_alias is an optional, secondary identifier specified by
    # the client.  This means for graph with name alias, we can operate the
    # graph resource by either id or alias, e.g.:
    #   GET /graphs/<GraphResource.id>
    #   GET /graphs/<name_alias>
    name_alias: Optional[str] = None


# Method: Validate ============================================================


@dataclass(config=_FORBID_EXTRA_CONFIG)
class GraphValidationRequest:
    # NOTE Response is of type common.ValidationResponse
    graph_definition: GraphDefinition
