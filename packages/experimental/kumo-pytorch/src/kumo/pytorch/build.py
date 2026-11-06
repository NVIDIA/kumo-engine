# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object-linked authoring of the same typed data-loader contract.

Tables expose typed columns as attributes. Paths refer to declared Link objects;
inputs refer to declared Events or Neighborhood objects. All encodings and
layouts are the ordinary public contract objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kumo.pytorch.contract.batch import Batching, BatchSpec
from kumo.pytorch.contract.batch import Input as CoreInput
from kumo.pytorch.contract.batch.sources import EntitySource, EventSource, NeighborhoodSource
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import (
    Aggregate,
    EntitySpec,
    EventSpec,
    Field,
    GraphLayer,
    GraphSample,
    NeighborhoodSpec,
    Rows,
    SampleSpec,
    TimeWindow,
)
from kumo.pytorch.contract.schema import (
    NON_TEMPORAL,
    DataSchema,
    NonTemporal,
    RelationSpec,
    TableSpec,
)
from kumo.pytorch.spec import DataLoaderSpec

__all__ = [
    "NON_TEMPORAL",
    "Batch",
    "Batching",
    "Column",
    "Count",
    "CountDistinct",
    "Events",
    "Hop",
    "Input",
    "Layer",
    "Link",
    "Max",
    "Mean",
    "Min",
    "Mode",
    "Neighborhood",
    "Read",
    "Recency",
    "Sample",
    "Schema",
    "StdPopulation",
    "StdSample",
    "Sum",
    "Table",
    "VariancePopulation",
    "VarianceSample",
    "contract",
]


@dataclass(frozen=True)
class Column:
    """A declared column carrying its table and stored type."""

    table: str
    name: str
    dtype: DType | None


@dataclass(frozen=True)
class Table:
    """A table with typed columns and an explicitly declared clock.

    Columns are available as attributes or as ``table["column"]``; an
    undeclared column is an ``AttributeError``. As with `TableSpec`, omit
    ``columns`` for Parquet discovery or use ``None`` for an inferred type.

    :param key: the primary key.
    :param time: the clock column or ``NON_TEMPORAL``; required.
    """

    name: str
    key: str
    columns: Mapping[str, DType | None] | None = None
    time: str | NonTemporal = field(kw_only=True)

    def __post_init__(self) -> None:
        specification = self.lower()
        object.__setattr__(self, "columns", specification.columns)

    def __getattr__(self, name: str) -> Column:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]

    def __getitem__(self, name: str) -> Column:
        columns = object.__getattribute__(self, "columns")
        if columns is not None and name not in columns:
            raise AttributeError(f"table {self.name!r} has no declared column {name!r}")
        return Column(table=self.name, name=name, dtype=None if columns is None else columns[name])

    def lower(self) -> TableSpec:
        """The equivalent `TableSpec`."""
        return TableSpec(
            name=self.name,
            primary_key=self.key,
            columns=self.columns,
            time_column=self.time,
        )


@dataclass(frozen=True)
class Link:
    """An equality relationship between declared columns."""

    name: str
    left: Column
    right: Column

    def lower(self) -> RelationSpec:
        """The equivalent `RelationSpec`."""
        return RelationSpec(
            name=self.name,
            left_table=self.left.table,
            left_key=self.left.name,
            right_table=self.right.table,
            right_key=self.right.name,
        )


@dataclass(frozen=True)
class Schema:
    """Named tables and links; their declaration placement has no meaning."""

    tables: tuple[Table, ...]
    links: tuple[Link, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tables", tuple(self.tables))
        object.__setattr__(self, "links", tuple(self.links))
        self.lower()

    def lower(self) -> DataSchema:
        """The equivalent `DataSchema`."""
        return DataSchema(
            tables=tuple(table.lower() for table in self.tables),
            relations=tuple(link.lower() for link in self.links),
        )

    def table(self, name: str) -> Table:
        """The declared table called ``name``; ``KeyError`` when there is none."""
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(f"schema has no table {name!r}")

    def destination(self, start: str, path: Sequence[Link]) -> str:
        """The table ``path`` reaches from ``start``.

        :raises ValueError: for an undeclared link or a disconnected path.
        """
        current = start
        declared = {link.name: link for link in self.links}
        for link in path:
            if declared.get(link.name) != link:
                raise ValueError(f"path uses undeclared link {link.name!r}")
            if current == link.left.table:
                current = link.right.table
            elif current == link.right.table:
                current = link.left.table
            else:
                raise ValueError(f"link {link.name!r} does not connect to table {current!r}")
        return current


@dataclass(frozen=True)
class Read:
    """Read a typed column without choosing its model encoding.

    The column's declared type is inherited, not repeated.
    """

    column: Column

    def lower(self, schema: Schema, table: str) -> Field:
        """The equivalent `Field`, read from ``table``."""
        if self.column.table != table:
            raise ValueError(
                f"read {self.column.table}.{self.column.name} is outside table {table!r}"
            )
        if schema.table(table)[self.column.name].dtype != self.column.dtype:
            raise ValueError(f"read {table}.{self.column.name} has an inconsistent declared type")
        return Field(column=self.column.name)


@dataclass(frozen=True)
class _Reduction:
    operation: str
    via: tuple[Link, ...]
    dtype: DType | None
    column: Column | None
    since: int | None
    until: int
    distinct: bool

    def lower(self, schema: Schema, table: str) -> Aggregate:
        destination = schema.destination(table, self.via)
        if self.column is not None and self.column.table != destination:
            raise ValueError(
                f"reduction {self.operation!r} reads {self.column.table!r}, "
                f"but its path ends at {destination!r}"
            )
        return Aggregate(
            source=_rows(
                schema=schema,
                table=table,
                via=self.via,
                since=self.since,
                until=self.until,
                distinct=self.distinct,
            ),
            operation=self.operation,
            column=None if self.column is None else self.column.name,
            dtype=self.dtype,
        )


def _reduction(operation: str, *, needs_column: bool):
    def constructor(
        column: Column | None = None,
        *,
        via: Link | Sequence[Link],
        dtype: DType | None = None,
        since: int | None = None,
        until: int = 0,
        distinct: bool = True,
    ) -> _Reduction:
        if needs_column != (column is not None):
            requirement = "requires" if needs_column else "does not accept"
            raise ValueError(f"{operation} {requirement} a value column")
        return _Reduction(
            operation=operation,
            via=_path(via),
            dtype=dtype,
            column=column,
            since=since,
            until=until,
            distinct=distinct,
        )

    constructor.__name__ = operation
    constructor.__doc__ = (
        f"``{operation}`` over the rows reached by ``via``, lowered to an `Aggregate`.\n\n"
        ":param via: a declared `Link`, or a path of them, from the sample entity.\n"
        ":param dtype: optional checked result type; inferred when omitted.\n"
        ":param since: exclusive window start in seconds from the anchor; open when None.\n"
        ":param until: inclusive window end in seconds from the anchor; the destination "
        "clock is inherited.\n"
        ":param distinct: keep one row per destination primary key."
    )
    return constructor


Count = _reduction("count", needs_column=False)
CountDistinct = _reduction("count_distinct", needs_column=True)
Sum = _reduction("sum", needs_column=True)
Mean = _reduction("mean", needs_column=True)
Min = _reduction("min", needs_column=True)
Max = _reduction("max", needs_column=True)
Mode = _reduction("mode", needs_column=True)
StdPopulation = _reduction("std_population", needs_column=True)
StdSample = _reduction("std_sample", needs_column=True)
VariancePopulation = _reduction("var_population", needs_column=True)
VarianceSample = _reduction("var_sample", needs_column=True)
Recency = _reduction("recency", needs_column=True)


@dataclass(frozen=True)
class Events:
    """Chronological event values selected through a declared link path.

    The destination table must declare a clock; ``since`` and ``until`` bound
    the window in seconds from the anchor. Fields may be `Read` objects or
    ordinary calculation objects.
    """

    via: Link | Sequence[Link]
    fields: Mapping[str, Any]
    limit: int | None = None
    order: str = "ascending"
    since: int | None = None
    until: int = 0

    def lower(self, schema: Schema, table: str) -> EventSpec:
        """The equivalent `EventSpec`, starting from ``table``."""
        path = _path(self.via)
        destination = schema.destination(table, path)
        rows = _rows(
            schema=schema,
            table=table,
            via=path,
            since=self.since,
            until=self.until,
            distinct=True,
        )
        if rows.time is None:
            raise ValueError(f"event table {destination!r} must declare a clock")
        return EventSpec(
            rows=rows,
            fields={
                name: value.lower(schema, destination) if isinstance(value, Read) else value
                for name, value in self.fields.items()
            },
            limit=self.limit,
            order=self.order,
        )


@dataclass(frozen=True)
class Hop:
    """One explicitly configured graph traversal.

    Inherits the destination clock unless ``time`` overrides it or
    ``temporal="static"`` disables filtering; see `GraphSample`.
    """

    link: Link
    fanout: int
    strategy: str = "recent"
    direction: str = "forward"
    time: str | None = None
    temporal: str = "inherit"

    def lower(self) -> GraphSample:
        """The equivalent `GraphSample`."""
        return GraphSample(
            relation=self.link.name,
            fanout=self.fanout,
            direction=self.direction,
            strategy=self.strategy,
            time_column=self.time,
            temporal=self.temporal,
        )


@dataclass(frozen=True)
class Layer:
    """Hops expanded together from one frontier."""

    hops: tuple[Hop, ...]

    def lower(self) -> GraphLayer:
        """The equivalent `GraphLayer`."""
        return GraphLayer(samples=tuple(hop.lower() for hop in self.hops))


@dataclass(frozen=True)
class Neighborhood:
    """A graph neighborhood: `Hop` objects, or explicit `Layer` groups, in order.

    An input must refer to this exact object as registered in the `Sample`; an
    equal, unregistered copy is refused.
    """

    layers: tuple[Layer | Hop, ...]

    def lower(self) -> NeighborhoodSpec:
        """The equivalent `NeighborhoodSpec`."""
        return NeighborhoodSpec(
            layers=tuple(
                value.lower() if isinstance(value, Layer) else GraphLayer(samples=(value.lower(),))
                for value in self.layers
            ),
        )


@dataclass(frozen=True, kw_only=True)
class Sample:
    """Point-in-time sampled values, using the ordinary calculation objects."""

    entity: Table
    entity_type: str | None = None
    values: Mapping[str, Any] = field(default_factory=dict)
    events: Mapping[str, Events] = field(default_factory=dict)
    neighbors: Mapping[str, Neighborhood] = field(default_factory=dict)
    dated_values: Mapping[str, Any] = field(default_factory=dict)
    targets: Mapping[str, Any] = field(default_factory=dict)
    candidate_sets: Mapping[str, Any] = field(default_factory=dict)

    def lower(self, schema: Schema) -> SampleSpec:
        """The equivalent `SampleSpec`."""
        table = self.entity.name
        return SampleSpec(
            entity=EntitySpec(table=table, entity_type=self.entity_type),
            entity_values={
                name: value.lower(schema, table) if isinstance(value, (Read, _Reduction)) else value
                for name, value in self.values.items()
            },
            events={name: value.lower(schema, table) for name, value in self.events.items()},
            neighborhoods={name: value.lower() for name, value in self.neighbors.items()},
            dated_entity_values=dict(self.dated_values),
            targets=dict(self.targets),
            candidate_sets=dict(self.candidate_sets),
        )


@dataclass(frozen=True)
class Input:
    """Bind an object-linked source to ordinary representation and layout objects."""

    source: Any
    representation: Any
    layout: Any

    def lower(self, names: _Names) -> CoreInput:
        """The equivalent core `Input`."""
        source = self.source
        if isinstance(source, Table):
            if source.name != names.sample.entity.name:
                raise ValueError(f"input entity {source.name!r} differs from the sample entity")
            source = EntitySource()
        elif isinstance(source, Events):
            source = EventSource(events=(names.event(source),))
        elif isinstance(source, Neighborhood):
            source = NeighborhoodSource(name=names.neighborhood(source))
        elif isinstance(source, (tuple, list)) and all(isinstance(item, Events) for item in source):
            source = EventSource(events=tuple(names.event(item) for item in source))
        return CoreInput(source=source, representation=self.representation, layout=self.layout)


@dataclass(frozen=True)
class Batch:
    """Named inputs and the same shared resources as BatchSpec."""

    inputs: Mapping[str, Input | CoreInput]
    targets: Mapping[str, Any] = field(default_factory=dict)
    encodings: Mapping[str, Any] = field(default_factory=dict)
    id_spaces: Mapping[str, Any] = field(default_factory=dict)
    field_names: Mapping[str, Any] = field(default_factory=dict)
    batching: Batching = field(default_factory=Batching)

    def lower(self, names: _Names) -> BatchSpec:
        """The equivalent `BatchSpec`."""
        return BatchSpec(
            inputs={
                name: value.lower(names) if isinstance(value, Input) else value
                for name, value in self.inputs.items()
            },
            targets=self.targets,
            encodings=self.encodings,
            id_spaces=self.id_spaces,
            field_names=self.field_names,
            batching=self.batching,
        )


class _Names:
    def __init__(self, sample: Sample):
        self.sample = sample
        self.events = {id(value): name for name, value in sample.events.items()}
        self.neighbors = {id(value): name for name, value in sample.neighbors.items()}

    def event(self, value: Events) -> str:
        if id(value) not in self.events:
            raise ValueError("input event source must be declared in Sample.events")
        return self.events[id(value)]

    def neighborhood(self, value: Neighborhood) -> str:
        if id(value) not in self.neighbors:
            raise ValueError("input neighborhood must be declared in Sample.neighbors")
        return self.neighbors[id(value)]


def contract(schema: Schema, sample: Sample, batch: Batch) -> DataLoaderSpec:
    """Produce the canonical typed contract; the engine executes only that value."""

    return DataLoaderSpec(
        schema=schema.lower(), sample=sample.lower(schema), batch=batch.lower(_Names(sample))
    )


def _path(value: Link | Sequence[Link]) -> tuple[Link, ...]:
    return (value,) if isinstance(value, Link) else tuple(value)


def _rows(
    *,
    schema: Schema,
    table: str,
    via: Sequence[Link],
    since: int | None,
    until: int,
    distinct: bool,
) -> Rows:
    destination = schema.table(schema.destination(table, via))
    clock = destination.time
    if isinstance(clock, NonTemporal):
        if since is not None or until != 0:
            raise ValueError(f"non-temporal table {destination.name!r} has no time window")
        window = None
    else:
        window = TimeWindow(column=clock, start=since, stop=until)
    return Rows(path=tuple(link.name for link in via), time=window, distinct=distinct)
