# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which rows a request needs, derived from the contract alone.

`ServingRowsSpec` describes, per table, the rows serving must be given for one
request: how they join to the request or to a previous hop, the time window, how
many the contract keeps, and which columns it reads. It never fetches anything.

The description is exact where the contract fixes the rows and conservative
where it does not: a reader that needs every matching row (a count, a uniform
graph sample, a lookup that must see every candidate) widens its table to ALL
matching rows, and the history line names that reader. A filter, an active
interval or `Rows.distinct` never narrows what is described; serving applies
them to the rows supplied. Columns only a target reads, and rows only a target
reads, are listed as not needed.

Import these types from `kumo.pytorch.serving`.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from kumo.pytorch.contract._frozen import FrozenMapping
from kumo.pytorch.contract.batch import (
    AlignedEventTarget,
    BatchSpec,
    CandidateSource,
    CandidateTarget,
    EntitySource,
    EventsOf,
    EventSource,
    Fields,
    Graph,
    LabelTarget,
    NeighborhoodSource,
    OccurrencesOf,
    StructuredTarget,
)
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import (
    Aggregate,
    Compare,
    Constant,
    DatedEntityValue,
    Derived,
    EntitySpec,
    EventSpec,
    EventTimeDelta,
    ExternalTarget,
    Field,
    GroupBy,
    Rows,
    RowSource,
    RowTarget,
    SampleSpec,
    TimeWindow,
)
from kumo.pytorch.contract.schema import RelationSpec, TableSpec, table_clock
from kumo.pytorch.spec import DataLoaderSpec, sample_without_targets

__all__ = [
    "ColumnRole",
    "ColumnRows",
    "RowLimit",
    "RowsJoin",
    "ServingRowsSpec",
    "TableRoute",
    "TableRows",
]

ColumnRole = Literal["primary key", "join key", "time column", "read"]
_ROLE_ORDER: dict[ColumnRole, int] = {"primary key": 0, "join key": 1, "time column": 2, "read": 3}
_WIDTH = 79
_NBSP = "\u00a0"


@dataclass(frozen=True)
class RowsJoin:
    """How one table's rows match a request: a column equal to a request field or a hop's column.

    :param column: this table's column that is matched.
    :param relation: the relation followed; ``None`` for a table keyed by the request itself.
    :param request_field: ``"entity_id"``, or ``"candidates[<set>]"`` for a candidate's
        ``entity_id``, when the rows are keyed by the request.
    :param from_table: the table of the previous hop, when the rows hang off it.
    :param from_column: that table's column the rows match.
    :param from_hop: the previous hop's number; the request's own row is hop 0.
    """

    column: str
    relation: str | None = None
    request_field: str | None = None
    from_table: str | None = None
    from_column: str | None = None
    from_hop: int | None = None


@dataclass(frozen=True)
class RowLimit:
    """How many matching rows a route keeps, when the contract keeps fewer than all.

    Rows tied with the last kept row on ``time_column`` are kept by primary key,
    so fetch every row tied at the boundary.

    :param count: rows kept per ``per``.
    :param order: ``"newest"`` or ``"oldest"`` by ``time_column``.
    :param per: ``"request"``, ``"candidate"``, or ``"<table> row"`` for each row of the
        table the rows are reached from (the previous hop for a graph).
    :param reader: the reader whose limit this is.
    """

    count: int
    order: Literal["newest", "oldest"]
    time_column: str
    per: str
    reader: str


@dataclass(frozen=True)
class TableRoute:
    """One way a table's rows are reached from a request.

    :param hop: relations followed from the request's own row (hop 0).
    :param join: how the rows match the request or the previous hop.
    :param time_column: the column the time window applies to; ``None`` when the
        rows are needed at any time.
    :param window: ``None`` for every row at or before the request's
        ``as_of_time`` (or at any time, when ``time_column`` is ``None``); otherwise
        the union of every reader's window, relative to ``as_of_time``.
    :param limit: ``None`` when every matching row is needed.
    :param history: why: the reader that needs this many rows, in words.
    :param read_by: the sample components that read these rows.
    """

    hop: int
    join: RowsJoin
    time_column: str | None
    window: TimeWindow | None
    limit: RowLimit | None
    history: str
    read_by: tuple[str, ...]


@dataclass(frozen=True)
class ColumnRows:
    """One declared column: its type, why serving needs it, and whether it may be omitted.

    :param role: ``"primary key"``, ``"join key"``, ``"time column"`` or ``"read"``;
        ``None`` for a column inference never reads.
    :param required: every supplied row must carry it; an optional column may be
        left out and reads as ``None``.
    :param read_by: the batch inputs (or, where no input encodes them, the sample
        components) whose values read it.
    """

    name: str
    dtype: DType
    role: ColumnRole | None
    required: bool
    read_by: tuple[str, ...]


@dataclass(frozen=True)
class TableRows:
    """The rows of one table a request needs.

    :param required: the table must be supplied, even as an empty list. An optional
        table is never read under inference and may be left out.
    :param routes: each way the table is reached, usually one; fetch their union.
    :param columns: every declared column, by name.
    """

    name: str
    required: bool
    routes: tuple[TableRoute, ...]
    columns: Mapping[str, ColumnRows]
    read_by: tuple[str, ...]

    @property
    def route(self) -> TableRoute:
        """The table's only route; refused when it has none or several (read ``routes``)."""

        if len(self.routes) != 1:
            raise ValueError(
                f"{self.name} is reached by {len(self.routes)} routes; read TableRows.routes"
            )
        return self.routes[0]

    @property
    def join(self) -> RowsJoin:
        return self.route.join

    @property
    def hop(self) -> int:
        return self.route.hop

    @property
    def time_column(self) -> str | None:
        return self.route.time_column

    @property
    def window(self) -> TimeWindow | None:
        return self.route.window

    @property
    def limit(self) -> RowLimit | None:
        return self.route.limit


@dataclass(frozen=True)
class ServingRowsSpec:
    """What to fetch per request for one contract; ``str(spec)`` renders it for reading.

    Derived from the contract alone; it fetches nothing. Per table it gives how
    rows match the request (a key equal to ``entity_id``, or ``IN`` the keys of
    the previous graph hop), the time window relative to ``as_of_time``, how
    many rows are kept and which reader decides, and each column's type and
    role. It is exact where the contract fixes the rows and conservative where
    it does not: filters, active intervals and ``Rows.distinct`` never narrow
    it; a path's intermediate tables are needed at any time; a uniform graph
    sample needs every candidate; a ``Field(path=...)`` lookup needs every
    related row up to the cutoff, except a ``selection="latest"`` lookup as of
    the sample time. Primary keys, join keys and time columns stay required.

    :param entity_table: the table a request's ``entity_id`` keys.
    :param tables: every declared table, required or not.
    :param not_needed: what the contract declares but inference never reads.
    """

    entity_table: str
    tables: Mapping[str, TableRows]
    not_needed: tuple[str, ...]

    @property
    def required_tables(self) -> tuple[str, ...]:
        """Tables a request must name, even with no rows (as an empty list)."""
        return tuple(name for name, table in self.tables.items() if table.required)

    def __str__(self) -> str:
        return _render(self)

    @classmethod
    def from_contract(cls, contract: DataLoaderSpec) -> ServingRowsSpec:
        """Derive the spec from a resolved contract's inference reads."""

        return _derive(contract)


# -- derivation ---------------------------------------------------------------


@dataclass(frozen=True)
class _Hop:
    relation: str
    from_table: str
    from_column: str
    table: str
    column: str


@dataclass(frozen=True)
class _Route:
    root: str
    root_table: str
    hops: tuple[_Hop, ...] = ()

    @property
    def table(self) -> str:
        return self.hops[-1].table if self.hops else self.root_table

    def then(self, hop: _Hop) -> _Route:
        return _Route(self.root, self.root_table, (*self.hops, hop))

    @property
    def per(self) -> str:
        if self.hops:
            return f"{self.table} row"
        return "request" if self.root == "entity_id" else "candidate"


_Time = tuple[str, int | None, int | None]


@dataclass(frozen=True)
class _Read:
    route: _Route
    origin: str
    unlimited: str
    noun: str
    time: _Time | None
    limit: RowLimit | None = None
    limited: str = ""

    @property
    def only(self) -> str:
        return self.limited.replace("keeps ", "keeps only ", 1).replace(
            "looks up ", "looks up only ", 1
        )


_VERBS = {
    "count": ("counts", "count"),
    "count_distinct": ("counts distinct values in", "distinct count"),
    "sum": ("sums", "sum"),
    "mean": ("averages", "mean"),
    "min": ("takes the minimum of", "minimum"),
    "max": ("takes the maximum of", "maximum"),
    "mode": ("takes the mode of", "mode"),
    "recency": ("takes the latest time in", "recency"),
    "var_population": ("takes the variance of", "variance"),
    "var_sample": ("takes the variance of", "variance"),
    "std_population": ("takes the standard deviation of", "standard deviation"),
    "std_sample": ("takes the standard deviation of", "standard deviation"),
}


class _Reads:
    """Every row and column one part of a contract reads, by symbolic route."""

    def __init__(self, contract: DataLoaderSpec, names: _BatchNames) -> None:
        self.schema = contract.schema
        self.names = names
        self.tables: dict[str, TableSpec] = {table.name: table for table in self.schema.tables}
        self.relations: dict[str, RelationSpec] = {
            relation.name: relation for relation in self.schema.relations
        }
        self.reads: list[_Read] = []
        self.columns: dict[tuple[str, str], dict[ColumnRole, set[str]]] = {}
        self.roots: list[_Route] = []

    def column(
        self, table: str, column: str, role: ColumnRole, readers: tuple[str, ...] = ()
    ) -> None:
        self.columns.setdefault((table, column), {}).setdefault(role, set()).update(readers)

    def root(self, root: str, entity: EntitySpec, scope: tuple[str, ...]) -> _Route:
        route = _Route(root, entity.table)
        if route not in self.roots:
            self.roots.append(route)
        table = self.tables[entity.table]
        self.column(entity.table, table.primary_key, "primary key")
        for condition in entity.filters:
            self.column(entity.table, condition.column, "read", (f"{_scope_path(scope)}.filters",))
        if entity.available_from is not None:
            self.column(
                entity.table,
                entity.available_from,
                "read",
                (f"{_scope_path(scope)}.available_from",),
            )
        return route

    def hop(self, route: _Route, relation_name: str, *, reverse: bool | None = None) -> _Route:
        relation = self.relations[relation_name]
        current = route.table
        forward = current == relation.left_table if reverse is None else not reverse
        if forward:
            hop = _Hop(
                relation.name,
                relation.left_table,
                relation.left_key,
                relation.right_table,
                relation.right_key,
            )
        else:
            hop = _Hop(
                relation.name,
                relation.right_table,
                relation.right_key,
                relation.left_table,
                relation.left_key,
            )
        self.column(hop.from_table, hop.from_column, "join key")
        self.column(hop.table, hop.column, "join key")
        self.column(hop.table, self.tables[hop.table].primary_key, "primary key")
        return route.then(hop)

    def follow(self, route: _Route, path: tuple[str, ...], origin: str) -> _Route:
        """Follow ``path``; every intermediate table is read at any time, in full."""

        for index, relation in enumerate(path):
            route = self.hop(route, relation)
            if index < len(path) - 1:
                destination = self._destination(route, path[index + 1 :])
                self.reads.append(
                    _Read(
                        route,
                        origin,
                        f"{origin} joins through every one of them to reach {destination}",
                        "join",
                        None,
                    )
                )
        return route

    def _destination(self, route: _Route, path: tuple[str, ...]) -> str:
        table = route.table
        for name in path:
            relation = self.relations[name]
            table = relation.right_table if table == relation.left_table else relation.left_table
        return table

    def rows(
        self,
        base: _Route,
        rows: Rows,
        origin: str,
        *,
        unlimited: str,
        noun: str,
        limit: tuple[int, Literal["newest", "oldest"]] | None = None,
        limited: str = "",
    ) -> _Route:
        """Record a `Rows` selection; returns the route of its destination rows."""

        route = self.follow(base, rows.path, origin)
        table = route.table
        for condition in rows.filters:
            self.column(table, condition.column, "read", (f"{origin} filter",))
        if rows.active is not None:
            for column in (rows.active.start_column, rows.active.end_column):
                self.column(table, column, "read", (f"{origin} active interval",))
        if rows.time is not None:
            clock: str | None = rows.time.column
            time: _Time | None = (rows.time.column, rows.time.start, rows.time.stop)
        else:
            clock = table_clock(self.tables[table])
            time = None if clock is None else (clock, None, 0)
        if clock is not None:
            self.column(table, clock, "time column")
        if not rows.path:
            return route
        row_limit = None
        if limit is not None and clock is not None:
            if rows.filters or rows.active is not None:
                unlimited = (
                    f"{origin} keeps the {limit[1]} {limit[0]} rows that pass its filters, "
                    "so it needs every row"
                )
            else:
                row_limit = RowLimit(limit[0], limit[1], clock, base.per, origin)
        self.reads.append(_Read(route, origin, unlimited, noun, time, row_limit, limited))
        return route

    def source(
        self,
        base: _Route,
        source: RowSource,
        aggregate: Aggregate,
        origin: str,
        readers: tuple[str, ...],
    ) -> None:
        """Record an aggregate's row source and the table columns it reduces."""

        verb, noun = _VERBS[aggregate.operation]
        groups: list[GroupBy] = []
        while isinstance(source, GroupBy):
            groups.append(source)
            source = source.source
        route = self.rows(
            base,
            source,
            origin,
            unlimited=f"{origin} {verb} {{span}}",
            noun=noun,
        )
        if groups:
            innermost = groups[-1]
            for key in innermost.keys:
                self.column(route.table, key, "read", readers)
            for reduction in innermost.values.values():
                if reduction.column is not None:
                    self.column(route.table, reduction.column, "read", readers)
        elif aggregate.column is not None:
            self.column(route.table, aggregate.column, "read", readers)

    def field(
        self,
        base: _Route,
        field: Field,
        origin: str,
        readers: tuple[str, ...],
    ) -> None:
        if not field.path:
            self.column(base.table, field.column, "read", readers)
            if field.time_column is not None:
                self.column(base.table, field.time_column, "time column")
            return
        route = self.follow(base, field.path, origin)
        table = route.table
        clock = field.time_column or table_clock(self.tables[table])
        if clock is not None:
            self.column(table, clock, "time column")
        row_limit = None
        if field.selection == "latest" and clock is not None and field.reference_time == "sample":
            row_limit = RowLimit(1, "newest", clock, base.per, origin)
        self.reads.append(
            _Read(
                route,
                origin,
                f"{origin} looks up every related row",
                "lookup",
                None if clock is None else (clock, None, 0),
                row_limit,
                "looks up the latest",
            )
        )
        self.column(table, field.column, "read", readers)

    def values(
        self,
        base: _Route,
        values: Mapping[str, Field | Constant | Aggregate | Compare | Derived | EventTimeDelta],
        scope: tuple[str, ...],
    ) -> None:
        for name, value in values.items():
            origin = f"{_scope_path(scope)}.{name}"
            readers = self.names.of(scope, name)
            if isinstance(value, Field):
                self.field(base, value, origin, readers)
            elif isinstance(value, Aggregate):
                self.source(base, value.source, value, origin, readers)
            elif isinstance(value, Compare):
                self.source(base, value.left.source, value.left, origin, readers)

    def events(
        self,
        base: _Route,
        events: Mapping[str, EventSpec],
        collection: Literal["events", "current_events"],
    ) -> None:
        for name, event in events.items():
            origin = f"{collection}.{name}"
            order: Literal["newest", "oldest"] = (
                "newest" if event.order == "descending" else "oldest"
            )
            route = self.rows(
                base,
                event.rows,
                origin,
                unlimited=f"{origin} keeps every event in {{span}}",
                noun="event history",
                limit=None if event.limit is None else (event.limit, order),
                limited=f"keeps the {order} {event.limit}",
            )
            self.values(route, event.fields, (collection, name))

    def dated(self, base: _Route, values: Mapping[str, DatedEntityValue]) -> None:
        for name, value in values.items():
            origin = f"dated_entity_values.{name}"
            route = self.rows(
                base,
                value.rows,
                origin,
                unlimited=f"{origin} reads every dated row in {{span}}",
                noun="dated values",
            )
            if isinstance(value.field, Field):
                self.field(route, value.field, origin, self.names.of(("entity_values",), name))
            self.values(route, value.fields, ("dated_entity_values", name))

    def neighborhood(self, base: _Route, name: str, spec_layers: Iterable) -> None:
        frontier = [base]
        for depth, layer in enumerate(spec_layers, start=1):
            following: list[_Route] = []
            origin = f"neighborhoods.{name} hop {depth}"
            for traversal in layer.samples:
                relation = self.relations[traversal.relation]
                start = (
                    relation.left_table
                    if traversal.direction == "forward"
                    else relation.right_table
                )
                for current in frontier:
                    if current.table != start:
                        continue
                    route = self.hop(
                        current, traversal.relation, reverse=traversal.direction == "reverse"
                    )
                    clock = traversal.time_column or table_clock(self.tables[route.table])
                    if clock is not None:
                        self.column(route.table, clock, "time column")
                    time = (
                        (clock, None, 0)
                        if clock is not None and traversal.temporal == "inherit"
                        else None
                    )
                    if traversal.strategy == "recent" and clock is not None:
                        read = _Read(
                            route,
                            origin,
                            "",
                            "sample",
                            time,
                            RowLimit(traversal.fanout, "newest", clock, current.per, origin),
                            f"keeps a recent fanout of {traversal.fanout}",
                        )
                    else:
                        read = _Read(
                            route,
                            origin,
                            f"{origin} samples {traversal.fanout} uniformly from every "
                            "matching row",
                            "uniform sample",
                            time,
                        )
                    self.reads.append(read)
                    if route not in following:
                        following.append(route)
            frontier = following


def _scope_path(scope: tuple[str, ...]) -> str:
    return ".".join(scope)


class _BatchNames:
    """Name a sample value by the batch inputs that encode it, or by its sample path."""

    def __init__(self, sample: SampleSpec, batch: BatchSpec) -> None:
        self.sample = sample
        self.batch = batch

    def _source(self, name: str):
        source = self.batch.inputs[name].source
        seen = {name}
        while isinstance(source, EventsOf) and source.input not in seen:
            seen.add(source.input)
            source = self.batch.inputs[source.input].source
        return source

    def _scope_values(self, scope: tuple[str, ...]) -> Mapping[str, object]:
        kind = scope[0]
        if kind == "entity_values":
            return {**self.sample.entity_values, **self.sample.dated_entity_values}
        if kind in {"events", "current_events"}:
            return getattr(self.sample, kind)[scope[1]].fields
        if kind == "dated_entity_values":
            return self.sample.dated_entity_values[scope[1]].fields
        if kind == "candidate_sets":
            return self.sample.candidate_sets[scope[1]].values
        return {}

    def _consumers(self, scope: tuple[str, ...], name: str) -> set[str]:
        values = self._scope_values(scope)
        found = {name}
        changed = True
        while changed:
            changed = False
            for other, value in values.items():
                if (
                    other not in found
                    and isinstance(value, Derived)
                    and any(item in found for item in value.inputs)
                ):
                    found.add(other)
                    changed = True
        return found

    def _reads(self, input_name: str, scope: tuple[str, ...]) -> bool:
        value = self.batch.inputs[input_name]
        source = self._source(input_name)
        kind = scope[0]
        if kind == "entity_values":
            return isinstance(value.source, EntitySource)
        if kind in {"events", "current_events"}:
            collection = "history" if kind == "events" else "current"
            return (
                isinstance(source, EventSource)
                and scope[1] in source.events
                and source.collection == collection
            )
        if kind == "dated_entity_values":
            if not isinstance(value.source, OccurrencesOf):
                return False
            parent = self.batch.inputs[value.source.input].representation
            return isinstance(parent, Fields) and any(
                use.field.split("/")[-1] == scope[1] for use in parent.fields
            )
        if kind == "candidate_sets":
            return isinstance(source, CandidateSource) and source.name == scope[1]
        return False

    def of(self, scope: tuple[str, ...], name: str) -> tuple[str, ...]:
        consumers = self._consumers(scope, name)
        found: list[str] = []
        for input_name, value in self.batch.inputs.items():
            if not isinstance(value.representation, Fields) or not self._reads(input_name, scope):
                continue
            selectors = [use.field for use in value.representation.fields]
            selectors += [
                use.field
                for target in self.batch.targets.values()
                if isinstance(target, AlignedEventTarget) and target.input == input_name
                for use in target.fields
            ]
            for selector in selectors:
                if selector.split("/")[-1] in consumers:
                    found.append(f"{input_name}.{selector}")
        return tuple(dict.fromkeys(found)) or (f"{_scope_path(scope)}.{name}",)

    def graph(self, neighborhood: str, table: str) -> dict[str, tuple[str, ...]]:
        found: dict[str, list[str]] = {}
        for input_name, value in self.batch.inputs.items():
            if (
                isinstance(value.source, NeighborhoodSource)
                and value.source.name == neighborhood
                and isinstance(value.representation, Graph)
                and table in value.representation.node_fields
            ):
                for use in value.representation.node_fields[table].fields:
                    found.setdefault(use.field, []).append(f"{input_name}.{use.field}")
        return {column: tuple(readers) for column, readers in found.items()}

    def target(self, name: str, candidate_set: str | None = None) -> str:
        labels = []
        for target_name, target in self.batch.targets.items():
            if candidate_set is None and isinstance(target, (LabelTarget, StructuredTarget)):
                if target.source == name:
                    labels.append(target_name)
            elif (
                candidate_set is not None
                and isinstance(target, CandidateTarget)
                and target.source == name
                and isinstance(self.batch.inputs[target.input].source, CandidateSource)
                and self.batch.inputs[target.input].source.name == candidate_set
            ):
                labels.append(target_name)
        if labels:
            return ", ".join(f'"{label}"' for label in labels)
        path = name if candidate_set is None else f"{candidate_set}.{name}"
        return f'"{path}"'


def _collect(contract: DataLoaderSpec, sample: SampleSpec, names: _BatchNames) -> _Reads:
    reads = _Reads(contract, names)
    entity = reads.root("entity_id", sample.entity, ("entity",))
    reads.values(entity, sample.entity_values, ("entity_values",))
    reads.dated(entity, sample.dated_entity_values)
    reads.events(entity, sample.events, "events")
    reads.events(entity, sample.current_events, "current_events")
    for name, neighborhood in sample.neighborhoods.items():
        reads.neighborhood(entity, name, neighborhood.layers)
        for table in reads.tables:
            for column, readers in names.graph(name, table).items():
                reads.column(table, column, "read", readers)
    for name, candidates in sample.candidate_sets.items():
        root = reads.root(
            f"candidates[{name!r}]", candidates.entity, ("candidate_sets", name, "entity")
        )
        reads.values(root, candidates.values, ("candidate_sets", name))
    return reads


def _target_reads(contract: DataLoaderSpec, names: _BatchNames) -> list[tuple[str, _Reads]]:
    sample = contract.sample
    found: list[tuple[str, _Reads]] = []

    def one(root: _Route, target: object, label: str, scope: tuple[str, ...], name: str) -> None:
        reads = _Reads(contract, names)
        reads.roots.append(root)
        origin = f"targets.{name}"
        if isinstance(target, RowTarget):
            route = reads.rows(root, target.rows, origin, unlimited="", noun="")
            reads.values(route, target.fields, scope)
        elif isinstance(target, Compare):
            reads.source(root, target.left.source, target.left, origin, (origin,))
        elif isinstance(target, Aggregate):
            reads.source(root, target.source, target, origin, (origin,))
        found.append((label, reads))

    entity = _Route("entity_id", sample.entity.table)
    for name, target in sample.targets.items():
        if not isinstance(target, ExternalTarget):
            one(entity, target, names.target(name), ("targets", name), name)
    for set_name, candidates in sample.candidate_sets.items():
        root = _Route(f"candidates[{set_name!r}]", candidates.entity.table)
        for name, target in candidates.targets.items():
            if not isinstance(target, ExternalTarget):
                one(root, target, names.target(name, set_name), ("targets", name), name)
    return found


def _merge_time(reads: list[_Read]) -> tuple[_Time | None, str]:
    if any(read.time is None for read in reads):
        return None, ""
    columns = {read.time[0] for read in reads if read.time is not None}
    if len(columns) > 1:
        return None, f"its readers filter on different time columns ({', '.join(sorted(columns))})"
    starts = [read.time[1] for read in reads if read.time is not None]
    stops = [read.time[2] for read in reads if read.time is not None]
    start = None if None in starts else min(s for s in starts if s is not None)
    stop = None if None in stops else max(s for s in stops if s is not None)
    return (columns.pop(), start, stop), ""


def _merge_limit(reads: list[_Read]) -> RowLimit | None:
    limits = [read.limit for read in reads]
    if any(limit is None for limit in limits):
        return None
    first = limits[0]
    assert first is not None
    times = {read.time for read in reads}
    for limit in limits:
        assert limit is not None
        if (limit.order, limit.time_column, limit.per) != (
            first.order,
            first.time_column,
            first.per,
        ):
            return None
    stops = {None if time is None else time[2] for time in times}
    if len(stops) > 1 or (first.order == "oldest" and len(times) > 1):
        return None
    return max(limits, key=lambda limit: limit.count if limit is not None else 0)


def _span(read: _Read) -> str:
    if read.time is None or read.time[1] is None:
        return "the full history"
    return "its window"


def _history(
    table: str, reads: list[_Read], time: _Time | None, limit: RowLimit | None, note: str
) -> str:
    if limit is not None:
        reader = next(read for read in reads if read.origin == limit.reader)
        return (
            f"{limit.order} {limit.count} matching rows per {limit.per} by "
            f"{table}.{limit.time_column}, with every row tied at the last one: "
            f"{limit.reader} {reader.limited}"
        )
    unlimited = [read for read in reads if read.limit is None]
    pool = unlimited or reads
    forcing = min(
        pool,
        key=lambda read: (
            0 if read.time is None or read.time[1] is None else 1,
            0 if read.time is None or read.time[1] is None else read.time[1],
        ),
    )
    if forcing.limit is None:
        reason = forcing.unlimited.replace("{span}", _span(forcing))
    else:
        others = [read for read in reads if read is not forcing and read.limit is not None]
        reason = (
            f"{forcing.origin} and {others[0].origin} keep different rows"
            if others
            else forcing.unlimited
        )
    limited = [read for read in reads if read.limit is not None and read is not forcing]
    if limited and forcing.limit is None:
        reason += (
            f" ({limited[0].origin} {limited[0].only}, but the {forcing.noun} needs every row)"
        )
    text = f"ALL matching rows: {reason}"
    windows = {read.time for read in reads if read.time is not None}
    if time is not None and time[1] is not None and len(windows) > 1:
        widest = min(
            (read for read in reads if read.time is not None),
            key=lambda read: read.time[1] if read.time and read.time[1] is not None else 0,
        )
        text += f"; the window is {widest.origin}'s, the widest"
    if note:
        text += f"; {note}, so rows are needed at any time"
    return text


def _route_join(route: _Route, pk: str) -> RowsJoin:
    if not route.hops:
        return RowsJoin(pk, request_field=route.root)
    hop = route.hops[-1]
    return RowsJoin(
        hop.column,
        relation=hop.relation,
        from_table=hop.from_table,
        from_column=hop.from_column,
        from_hop=len(route.hops) - 1,
    )


def _derive(contract: DataLoaderSpec) -> ServingRowsSpec:
    contract = contract.resolve()
    sample = contract.sample
    names = _BatchNames(sample, contract.batch)
    serving = _collect(contract, sample_without_targets(sample), names)
    targets = _target_reads(contract, names)
    by_route: dict[_Route, list[_Read]] = {}
    for read in serving.reads:
        by_route.setdefault(read.route, []).append(read)
    routes_by_table: dict[str, list[_Route]] = {}
    for route in [*serving.roots, *by_route]:
        bucket = routes_by_table.setdefault(route.table, [])
        if route not in bucket:
            bucket.append(route)

    tables: dict[str, TableRows] = {}
    for spec in sorted(
        contract.schema.tables,
        key=lambda table: (
            min((len(r.hops) for r in routes_by_table.get(table.name, [])), default=99),
            table.name != sample.entity.table,
            table.name,
        ),
    ):
        routes = []
        for route in routes_by_table.get(spec.name, []):
            reads = by_route.get(route, [])
            if not route.hops:
                routes.append(
                    TableRoute(0, _route_join(route, spec.primary_key), None, None, None, "", ())
                )
                continue
            time, note = _merge_time(reads)
            limit = _merge_limit(reads)
            window = (
                None
                if time is None or (time[1] is None and time[2] == 0)
                else TimeWindow(time[0], time[1], time[2])
            )
            routes.append(
                TableRoute(
                    hop=len(route.hops),
                    join=_route_join(route, spec.primary_key),
                    time_column=None if time is None else time[0],
                    window=window,
                    limit=limit,
                    history=_history(spec.name, reads, time, limit, note),
                    read_by=tuple(dict.fromkeys(read.origin for read in reads)),
                )
            )
        required = bool(routes)
        columns: dict[str, ColumnRows] = {}
        read_by: list[str] = []
        for column, dtype in (spec.columns or {}).items():
            roles = serving.columns.get((spec.name, column), {}) if required else {}
            role = min(roles, key=lambda item: _ROLE_ORDER[item]) if roles else None
            readers = tuple(sorted({reader for found in roles.values() for reader in found}))
            read_by.extend(readers)
            columns[column] = ColumnRows(column, dtype, role, role is not None, readers)
        tables[spec.name] = TableRows(
            spec.name,
            required,
            tuple(routes),
            FrozenMapping(columns),
            tuple(dict.fromkeys(read_by)),
        )
    return ServingRowsSpec(sample.entity.table, FrozenMapping(tables), _not_needed(tables, targets))


def _not_needed(
    tables: Mapping[str, TableRows], targets: list[tuple[str, _Reads]]
) -> tuple[str, ...]:
    items: dict[str, list[str]] = {}
    future: dict[str, list[str]] = {}
    for label, reads in targets:
        for read in reads.reads:
            table = tables[read.route.table]
            if not table.required:
                items.setdefault(table.name, []).append(label)
                continue
            if read.time is None or read.time[1] is None:
                continue
            serving = table.routes
            if serving and all(
                route.time_column == read.time[0]
                and _stop(route) is not None
                and _stop(route) <= read.time[1]
                and route.hop > 0
                for route in serving
            ):
                future.setdefault(table.name, []).append(label)
        for (table_name, column), _roles in reads.columns.items():
            table = tables[table_name]
            if table.required and not table.columns[column].required:
                items.setdefault(f"{table_name}.{column}", []).append(label)
    lines = []
    for table_name, labels in future.items():
        prefix = "rows" if len(future) == 1 else f"{table_name} rows"
        lines.append(f"{prefix} after request.as_of_time (read only by {_targets(labels)})")
    for item, labels in items.items():
        lines.append(f"{item} (read only by {_targets(labels)})")
    for table in tables.values():
        if not table.required and table.name not in items:
            lines.append(f"{table.name} (nothing reads it under inference)")
    return tuple(dict.fromkeys(lines))


def _stop(route: TableRoute) -> int | None:
    return 0 if route.window is None else route.window.stop


def _targets(labels: list[str]) -> str:
    unique = list(dict.fromkeys(labels))
    noun = "target" if len(unique) == 1 and "," not in unique[0] else "targets"
    return f"{noun} {', '.join(unique)}, which inference never computes"


# -- rendering ----------------------------------------------------------------


def dtype_text(dtype: DType) -> str:
    """A declared type as messages write it: ``timestamp[s, UTC]``, ``decimal128(10, 2)``."""

    if dtype.name == "timestamp":
        return f"timestamp[{dtype.unit}, {dtype.timezone}]"
    if dtype.name.startswith("decimal"):
        return f"{dtype.name}({dtype.precision}, {dtype.scale})"
    return dtype.name


def _offset(seconds: int | None) -> str:
    if not seconds:
        return "request.as_of_time"
    sign = "+" if seconds > 0 else "-"
    return f"request.as_of_time {sign} {_duration(abs(seconds))}"


def _duration(seconds: int) -> str:
    for unit, size in (("d", 86_400), ("h", 3_600), ("m", 60)):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def _match_lines(table: TableRows, route: TableRoute, spec: ServingRowsSpec) -> list[str]:
    join = route.join
    relation = f"   (relation{_NBSP}{join.relation})" if join.relation else ""
    if join.request_field == "entity_id":
        lines = [f"{table.name}.{join.column} = request.entity_id"]
    elif join.request_field is not None:
        lines = [f"{table.name}.{join.column} = each request.{join.request_field} entity_id"]
    elif join.from_hop == 0 and join.from_table == spec.entity_table:
        source = spec.tables[join.from_table]
        root_key = next(
            (name for name, column in source.columns.items() if column.role == "primary key"),
            None,
        )
        if join.from_column == root_key:
            lines = [f"{table.name}.{join.column} = request.entity_id{relation}"]
        else:
            lines = [
                f"{table.name}.{join.column} = {join.from_table}.{join.from_column} of the "
                f"request's {join.from_table} row{relation}"
            ]
    else:
        prefix = f"hop {route.hop} · " if route.hop > 1 else ""
        lines = [
            f"{prefix}{table.name}.{join.column} IN ({join.from_table}.{join.from_column} "
            f"from hop {join.from_hop}){relation}"
        ]
    if route.time_column is not None:
        column = f"{table.name}.{route.time_column}"
        window = route.window
        start, stop = (None, 0) if window is None else (window.start, window.stop)
        upper = "" if stop is None else f"{column} <= {_offset(stop)}"
        if start is None:
            lines.append(upper)
        elif stop is None:
            lines.append(f"{column} > {_offset(start)}")
        else:
            lines.append(f"{_offset(start)} < {column} <= {_offset(stop)}")
    return lines


def _wrap(label: str, text: str, gutter: int) -> list[str]:
    return _fill(f"  {label}".ljust(gutter), text)


def _fill(prefix: str, text: str, indent: int | None = None) -> list[str]:
    """Wrap ``text`` after ``prefix``; a non-breaking space keeps a phrase on one line."""

    lines = textwrap.wrap(
        text,
        width=_WIDTH,
        initial_indent=prefix,
        subsequent_indent=" " * (len(prefix) if indent is None else indent),
        break_on_hyphens=False,
        break_long_words=False,
    )
    return [line.replace(_NBSP, " ") for line in lines]


def _readers(readers: tuple[str, ...]) -> str:
    shown = ", ".join(readers[:3])
    more = len(readers) - 3
    return f"read{_NBSP}by: {shown}" + (f" and {more} more" if more > 0 else "")


def _render(spec: ServingRowsSpec) -> str:
    gutter = max(13, *(len(name) + 2 for name in spec.tables))
    lines = [
        f"Rows to supply per request (entity: {spec.entity_table}, keyed by request.entity_id)",
    ]
    for table in spec.tables.values():
        if not table.required:
            continue
        lines.append("")
        if len(table.routes) == 1 and table.routes[0].hop == 0:
            per = "1 row per request" if table.name == spec.entity_table else "1 row per candidate"
        elif len(table.routes) == 1 and table.routes[0].limit is not None:
            limit = table.routes[0].limit
            per = f"0..{limit.count} rows per {limit.per}"
        else:
            per = "0..n rows per request"
        lines.append(f"{table.name.ljust(gutter)}required · {per}")
        for index, route in enumerate(table.routes):
            label = "match" if index == 0 else "or match"
            for position, text in enumerate(_match_lines(table, route, spec)):
                lines.extend(_wrap(label if position == 0 else "", text, gutter))
            if route.history:
                lines.extend(_wrap("history", route.history, gutter))
        required = sorted(
            (column for column in table.columns.values() if column.required),
            key=lambda column: (_ROLE_ORDER[column.role], column.name),
        )
        name_width = max(len(column.name) for column in required) + 2
        type_width = max(len(dtype_text(column.dtype)) for column in required) + 2
        for index, column in enumerate(required):
            describe = (
                _readers(column.read_by)
                if column.role == "read"
                else column.role
                if not column.read_by
                else f"{column.role} · {_readers(column.read_by)}"
            )
            head = f"  {'columns' if index == 0 else ''}".ljust(gutter)
            name = column.name.ljust(name_width)
            lines.extend(
                _fill(
                    f"{head}{name}{dtype_text(column.dtype).ljust(type_width)}",
                    describe,
                    gutter + 2,
                )
            )
        optional = [column.name for column in table.columns.values() if not column.required]
        if optional:
            lines.extend(
                _wrap(
                    "optional",
                    f"{', '.join(sorted(optional))}: not read under inference; omit them "
                    "or pass None",
                    gutter,
                )
            )
    if spec.not_needed:
        lines.append("")
        if len(spec.not_needed) == 1:
            lines.extend(
                textwrap.wrap(
                    f"Not needed for serving: {spec.not_needed[0]}",
                    width=_WIDTH,
                    break_on_hyphens=False,
                    break_long_words=False,
                )
            )
        else:
            lines.append("Not needed for serving:")
            for item in spec.not_needed:
                lines.extend(
                    textwrap.wrap(
                        item,
                        width=_WIDTH,
                        initial_indent="  - ",
                        subsequent_indent="    ",
                        break_on_hyphens=False,
                        break_long_words=False,
                    )
                )
    return "\n".join(lines)
