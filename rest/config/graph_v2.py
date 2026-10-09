# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass, field
from functools import cached_property
from hashlib import md5
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import kumoapi.data_snapshot as snapshot_api
import kumoapi.graph as api
from kumo_core.types.schema import FPID
from pydantic import (
    BaseModel,
    ValidationError,
    field_validator,
    model_validator,
)

from io_utils import kumo_dataclass
from io_utils.dataclasses import before_validator_dict
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.table_v2 import TableDefinition, TableKind
from statedb.config.common import (
    ColumnID,
    ConnectorID,
    DBEntry,
    GraphID,
    GraphIDv2,
    GraphLink,
    PredictiveQueryID,
    TableID,
    TableIDv2,
    TableName,
    to_table_id,
)
from statedb.config.table import TableDataVersion
from statedb.config.timestamp import ResourceTimestamps
from statedb.rest.graph import ColumnGroup
from statedb.rest.graph_v2 import GraphDBEntryV2 as SerializedGraphDBEntryV2


@kumo_dataclass(frozen=True)
class ColumnKey(api.ColumnKey):
    """Reference to a column within a PQuery definition."""
    @classmethod
    def from_kumoapi(cls, column_key: api.ColumnKey) -> 'ColumnKey':
        return cls(
            table_name=column_key.table_name,
            col_name=column_key.col_name,
        )

    def to_kumoapi(self) -> api.ColumnKey:
        return api.ColumnKey(
            table_name=self.table_name,
            col_name=self.col_name,
        )

    @cached_property
    def fpid(self) -> FPID:
        md5_hasher = md5()
        md5_hasher.update(self.table_name.encode())
        md5_hasher.update(self.col_name.encode())
        return FPID(md5_hasher.hexdigest())


def _coerce_column_key(value: Any) -> ColumnKey:
    if isinstance(value, ColumnKey):
        return value
    if isinstance(value, api.ColumnKey):
        return ColumnKey.from_kumoapi(value)
    if dataclasses.is_dataclass(value):
        return ColumnKey(**dataclasses.asdict(
            value))  # pyright: ignore[reportArgumentType]
    raise ValueError(f'Unsupported column key: {value}, type={type(value)}')


@kumo_dataclass(frozen=True)
class ColumnKeyGroup(api.ColumnKeyGroup):
    @field_validator('columns')
    @classmethod
    def check_dupe_and_sort(cls, v: Sequence) -> Tuple[ColumnKey, ...]:
        # This validator does two things to convert 'columns' field value:
        # 1) coerce/convert each column key element to "ColumnKey" type.
        # 2) dedupe and sort the column keys.
        columns = [_coerce_column_key(elem) for elem in v]

        # Dedup & sort by fpid
        col_dict = {col.fpid: col for col in columns}
        sorted_deduped = [col_dict[key] for key in sorted(col_dict)]
        if len(sorted_deduped) < len(v):
            num_dupes = len(v) - len(sorted_deduped)
            raise ValueError(
                f'All columns must be unique, found {num_dupes} dupes')
        return tuple(sorted_deduped)

    def _column_keys(self) -> Tuple[ColumnKey, ...]:
        return tuple(_coerce_column_key(col) for col in self.columns)

    @cached_property
    def fpid(self) -> FPID:
        md5_hasher = md5()
        for col in self._column_keys():
            md5_hasher.update(col.fpid.encode())
        return FPID(md5_hasher.hexdigest())

    def as_column_group(
            self, table_name_to_id: Dict[TableName, TableID]) -> ColumnGroup:
        """The data backend expects the ColumnGroup structure, which is the
        same as ColumnKeyGroup, but uses TableID instead of TableName.
        """
        cols: List[ColumnID] = []
        for col in self._column_keys():
            table_name = TableName(col.table_name)
            if table_name not in table_name_to_id:
                raise ValueError(
                    f"Table name is missing from the map: {table_name}")
            table_id = table_name_to_id[table_name]
            cols.append(ColumnID(table_id, col.col_name))
        return ColumnGroup(cols)


@kumo_dataclass
class GraphDefinition(api.GraphDefinition):
    @field_validator('tables')
    @classmethod
    def _coerce_tables(cls, tables: dict):
        # Coerce the 'tables' field to Dict[str, TableDefinition]
        tables_dict = {}
        for k, v in tables.items():
            if isinstance(v, TableDefinition):
                tables_dict[k] = v
            elif dataclasses.is_dataclass(v):
                tables_dict[k] = TableDefinition(**dataclasses.asdict(
                    v))  # pyright: ignore[reportArgumentType]
            elif isinstance(v, BaseModel):
                tables_dict[k] = TableDefinition(**dict(v))
            else:
                raise ValueError(
                    f'Unknown table definition (type={type(v)}): {v}')
        return tables_dict

    @field_validator('col_groups')
    @classmethod
    def _coerce_col_groups(cls, col_groups: List[api.ColumnKeyGroup]):
        # Coerce the 'col_groups' field to List[ColumnKeyGroup]
        return [ColumnKeyGroup(v.columns) for v in col_groups]

    def _tables(self) -> Dict[str, TableDefinition]:
        tables: Dict[str, TableDefinition] = {}
        for table_name, table_args in self.tables.items():
            if isinstance(table_args, TableDefinition):
                tables[str(table_name)] = table_args
            elif dataclasses.is_dataclass(table_args):
                tables[str(table_name)] = TableDefinition(
                    **dataclasses.asdict(
                        table_args))  # pyright: ignore[reportArgumentType]
            elif isinstance(table_args, BaseModel):
                tables[str(table_name)] = TableDefinition(**dict(table_args))
            else:
                raise ValueError(
                    f'Unknown table definition (type={type(table_args)}): '
                    f'{table_args}')
        return tables

    def _column_key_groups(self) -> List[ColumnKeyGroup]:
        return [
            col_group if isinstance(col_group, ColumnKeyGroup) else
            ColumnKeyGroup(col_group.columns) for col_group in self.col_groups
        ]

    @cached_property
    def fpid(self) -> FPID:
        # Generate a stable & unique fingerprint id based on the table
        # definition, discarding inferred information (such as dtype)
        md5_hasher = md5()
        for table_name, table_args in sorted(self._tables().items()):
            md5_hasher.update(table_name.encode())
            md5_hasher.update(table_args.fpid.encode())
        # Dedup & sort column group fingerpints
        cg_fps = sorted(set([cg.fpid for cg in self._column_key_groups()]))
        md5_hasher.update(','.join(cg_fps).encode())

        # "display_name" is EXCLUDED from fingerprinting.

        return FPID(md5_hasher.hexdigest())

    @property
    def edges(self) -> List[api.Edge]:
        """Re-generates individual edge structs from column key groups.

        Returns:
            List[api.Edge]: A list of all edges in the graph
        """
        # This legacy view is consumed by edge-health matching with its table
        # orientation reversed. Keep that presentation while sharing the
        # primary-key and foreign-key resolution decision.
        return [
            api.Edge(
                src_table=edge.dst_table,
                fkey=edge.fkey,
                dst_table=edge.src_table,
            ) for edge in api.resolve_foreign_keys(self.tables, self.col_groups)
        ]


def build_graph_definition(
    graph: Union[api.GraphDefinition, Dict[str, Any], BaseModel],
) -> GraphDefinition:
    """Coerce a public graph payload into the validated internal definition.

    Dataclass, dictionary, and Pydantic inputs are accepted. Nested Pydantic
    failures are projected into a sanitized ``ValidationException`` containing
    only field paths and validation messages.

    Raises:
        ValueError: If ``graph`` is not a supported input representation.
        ValidationException: If the graph definition fails validation.
    """
    from kumo.validation.validation import ValidationException

    if isinstance(graph, GraphDefinition):
        return graph
    if dataclasses.is_dataclass(graph) and not isinstance(graph, type):
        data = dataclasses.asdict(graph)
    elif isinstance(graph, dict):
        data = graph
    elif isinstance(graph, BaseModel):
        data = dict(graph)
    else:
        raise ValueError(f'Invalid GraphDefinition type ({type(graph)}),'
                         f' unable to coerce graph: {graph}')

    try:
        return GraphDefinition(**data)
    except ValidationError as err:
        details = []
        for error in err.errors():
            reason = error['msg']
            prefix = 'Value error, '
            if reason.startswith(prefix):
                reason = reason[len(prefix):]
            field_path = '.'.join(str(part) for part in error['loc'])
            details.append(f'{field_path}: {reason}' if field_path else reason)
        raise ValidationException(
            title='Graph Validation Failure',
            message='\n'.join(details),
        ) from err


@kumo_dataclass
class CanonicalGraphDBEntry(DBEntry):
    """A standard representation of GraphDBEntry and GraphDBEntryV2, making it
    easier to write generic code that works for both types of objects.
    """
    # ID of the entry in the database. Publicly exposed in the API.
    id: GraphID
    # global dedup key, graph name for v1; "NA-<fpid>" for v2
    # For internal use only, by data and ML backend
    dedup_key: str
    display_name: str
    table_name_to_id: Dict[TableName, TableID]
    col_groups: List[ColumnKeyGroup]
    display_description: str

    @property
    def column_groups(self) -> List[ColumnGroup]:
        column_groups: List[ColumnGroup] = []
        for grp in self.col_groups:
            column_groups.append(grp.as_column_group(self.table_name_to_id))
        return column_groups

    def to_graph_link(self) -> GraphLink:
        return GraphLink(name=self.display_name, id=self.id)


@kumo_dataclass
class GraphDBEntryV2(SerializedGraphDBEntryV2):
    @field_validator('col_groups')
    @classmethod
    def _coerce_col_groups(
        cls,
        col_groups: List[api.ColumnKeyGroup],
    ) -> List[ColumnKeyGroup]:
        return [ColumnKeyGroup(v.columns) for v in col_groups]

    def _column_key_groups(self) -> List[ColumnKeyGroup]:
        return [
            col_group if isinstance(col_group, ColumnKeyGroup) else
            ColumnKeyGroup(col_group.columns) for col_group in self.col_groups
        ]

    def to_graph_definition(self,
                            tables: List[TableDefinition]) -> GraphDefinition:
        tables_by_fpid = {t.fpid: t for t in tables}
        tables_by_name = {
            t_name: tables_by_fpid[t_fpid]
            for t_name, t_fpid in self.table_name_to_dbid.items()
        }
        return GraphDefinition(**{
            'tables': tables_by_name,
            'col_groups': self._column_key_groups(),
        })

    def as_canonical_entry(
            self,
            pquery_id: Optional[PredictiveQueryID]) -> CanonicalGraphDBEntry:
        graph_id = GraphIDv2.compose(pquery_id, self.id)
        tables: Dict[TableName, TableID] = {}
        for table_name, fpid in self.table_name_to_dbid.items():
            # Prefer to hide FPID in table id v2 when pquery_id is present.
            table_id = TableIDv2.compose(pquery_id, table_name,
                                         (None if pquery_id else fpid))
            tables[table_name] = table_id
        display_name = self.alias or (f'{pquery_id}_graph'
                                      if pquery_id else graph_id)
        return CanonicalGraphDBEntry(
            id=graph_id,
            dedup_key=GraphIDv2.compose(None, self.id),
            display_name=display_name,
            table_name_to_id=tables,
            col_groups=self._column_key_groups(),  # noqa: E126
            timestamps=self.timestamps,
            display_description=self.description)


@kumo_dataclass
class GraphResourceResponse:
    """
    A config object containing graph template metadata.
    Similar to api.GraphResource, but also includes
    timestamps.
    Used as response for `get_graph_v2`.
    """
    # Graph resource identifier in the format of "graph-<16digit hex id>",
    # this is generated on the backend.
    id: str

    # Immutable metadata definition for a graph.
    graph: api.GraphDefinition | GraphDefinition

    # Human-friendly name alias for the graph template.
    # Once a graph resource sets the name
    # alias, it cannot be renamed. Name aliases must be unique, in other words,
    # 1:1 mapping is required between graph name aliases to GraphResource.id
    # For graph with name alias, we can operate the
    # graph resource by either id or alias, e.g.:
    #   GET /graphs/<GraphResource.id>
    #   GET /graphs/<name_alias>
    name_alias: str | None

    # Created and last updated timestamps.
    timestamps: ResourceTimestamps

    table_name_to_fpid_mapping: Dict[TableName, FPID]
    num_tables: Optional[int] = None
    num_links: Optional[int] = None

    created_by: str = ""
    created_by_name: str = ""

    @model_validator(mode='before')
    @classmethod
    def _default_optional_fields(cls, data: Any) -> Any:
        # `name_alias` is optional but precedes required fields, so it cannot
        # carry an in-place default without reordering (which would change
        # serialization order). Inject the implicit `None` when omitted.
        data = before_validator_dict(data)
        if isinstance(data, dict) and 'name_alias' not in data:
            data['name_alias'] = None
        return data


@kumo_dataclass
class GraphSnapshotRequest:
    """
        A config object used to fetch graph snapshot.
        In the SDK, a training and a training table job can be triggered with different graphs.
        The UI should show the appropriate graph for each (currently we only show the graph snapshot for
        the training table generation job). The get_graph_snapshot endpoint takes graph definition and graph
        version as additional arguments. If those are supplied, the endpoint relies on them to fetch the graph snapshot.
        Otherwise, it only relies on the graph snapshot ID.
    """
    graph_snapshot_id: Optional[snapshot_api.GraphSnapshotID] = None
    graph_definiton: Optional[GraphDefinition] = None
    graph_version: Optional[str] = None


@kumo_dataclass
class GraphSnapshotResponse:
    snapshot_resource: snapshot_api.GraphSnapshotResource
    definition: api.GraphDefinition


@kumo_dataclass
class ListResponse:
    """
    A config object containing graph or table template metadata.
    Used as response for endpoint that lists graph/table templates.
    Currently medata includes alias (name), id, timestamps,
    source table name, and connector ID.
    """
    name: str
    id: Optional[str] = None
    timestamps: Optional[ResourceTimestamps] = None
    source_table_name: Optional[str] = None
    connector_id: Optional[ConnectorID] = None
    num_tables: Optional[int] = None
    num_links: Optional[int] = None
    created_by_name: str = ""
    # Raw-vs-Custom-SQL discriminator for table rows, stamped by the Tables-list
    # builder so the UI "Kind" filter classifies straight off the wire. Left
    # ``None`` for graph rows (the field is table-only) and, defensively, for any
    # table row a builder does not stamp; the UI treats absent as ``raw``.
    table_kind: Optional[TableKind] = None


@kumo_dataclass
class GraphSnapshotStatsResponse:
    """
    Response model for graph snapshot statistics.
    Contains aggregated statistics across all tables in the snapshot.
    """
    total_rows: Optional[int] = None
    total_size_bytes: Optional[int] = None


@dataclass
class GraphDataComputeInput:
    """Workflow-safe graph input containing only durable connector references."""
    graph_display_name: str
    graph_key: str
    tables: Dict[TableID, TableDefinition]
    base_tables: Dict[TableID, TableDefinition]
    connectors: Dict[ConnectorID, ConnectorExecutionRef]
    graph_definition: GraphDefinition
    table_aliases: Dict[TableID, TableName] = field(default_factory=dict)
    pinned_data_versions: Dict[TableID,
                               TableDataVersion] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for connector_id, reference in self.connectors.items():
            if not isinstance(reference, ConnectorExecutionRef):
                raise TypeError("GraphDataComputeInput connectors must be "
                                "ConnectorExecutionRef values")
            if ConnectorID(connector_id) != reference.connector_id:
                raise ValueError(
                    "GraphDataComputeInput connector key must match its "
                    "reference connector_id")
        project_ids = {
            reference.project_id
            for reference in self.connectors.values()
        }
        if len(project_ids) > 1:
            raise ValueError(
                "GraphDataComputeInput cannot contain cross-project "
                "connector references")

    @field_validator('tables')
    @classmethod
    def _coerce_table_ids(cls, v: Dict[str, Any]):
        return {to_table_id(k): v for k, v in v.items()}
