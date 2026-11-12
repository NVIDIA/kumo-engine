# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one entity is allowed to know at one prediction time.

`SampleSpec` and everything it composes: row selection and its time windows,
fields, aggregates, event sequences, graph neighborhoods, candidates and
targets. This layer is still raw-data territory — it chooses *rows and values*,
never token IDs, tensor shapes, padding or model layers. Those are
`kumo.pytorch.contract.batch`.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from dataclasses import field as dataclass_field
from decimal import Decimal, localcontext
from graphlib import CycleError, TopologicalSorter
from typing import Any, Literal, TypeAlias, get_args

from kumo.pytorch.contract._frozen import FrozenMapping
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.schema import DataSchema
from kumo.pytorch.data.records import (
    EntityHistory,
    Event,
    Scalar,
)

__all__ = [
    "ActiveInterval",
    "Aggregate",
    "BuiltSample",
    "CandidateRequest",
    "CandidateSetSpec",
    "Compare",
    "Constant",
    "CalendarPart",
    "ComparisonOperation",
    "DatedEntityValue",
    "Derived",
    "DerivedOperation",
    "EntitySpec",
    "EntityValue",
    "EventSpec",
    "EventTimeDelta",
    "ExternalTarget",
    "Field",
    "FilterOperation",
    "GraphLayer",
    "GraphSample",
    "GroupBy",
    "MapValues",
    "NeighborhoodSpec",
    "Reduce",
    "ReferenceTime",
    "RowFilter",
    "RowSource",
    "RowTarget",
    "Rows",
    "SampleRequest",
    "SampleSpec",
    "Selection",
    "SummaryOperation",
    "TargetValue",
    "TimeWindow",
]


def _mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return FrozenMapping(dict(value))


def _result_dtype(value: DType | None) -> None:
    if value is not None and not isinstance(value, DType):
        raise TypeError("calculation dtype must be a DType declaration")


Selection = Literal["one", "all", "latest"]
ReferenceTime = Literal["sample", "event"]
FilterOperation = Literal[
    "equal",
    "not_equal",
    "in",
    "not_in",
    "greater_than",
    "greater_or_equal",
    "less_than",
    "less_or_equal",
    "is_null",
    "not_null",
]
SummaryOperation = Literal[
    "count",
    "count_distinct",
    "sum",
    "mean",
    "min",
    "max",
    "std_population",
    "std_sample",
    "var_population",
    "var_sample",
    "recency",
    "mode",
]
#: Every reduction the contract can express.
ALL_REDUCTIONS: frozenset[str] = frozenset(get_args(SummaryOperation))

ComparisonOperation = Literal[
    "equal",
    "not_equal",
    "greater_than",
    "greater_or_equal",
    "less_than",
    "less_or_equal",
]
_ALL_COMPARISON_OPERATIONS = frozenset(get_args(ComparisonOperation))


DerivedOperation = Literal["add", "subtract", "multiply", "divide", "log1p", "abs", "sign"]


@dataclass(frozen=True)
class RowFilter:
    """One explicit comparison applied to rows selected by a recipe.

    :param value: the comparison operand; a sequence for ``in``/``not_in`` and
        unused by ``is_null``/``not_null``.
    """

    column: str
    operation: FilterOperation
    value: Any = None

    def __post_init__(self) -> None:
        if not self.column or self.operation not in get_args(FilterOperation):
            raise ValueError("a row filter requires a column and a supported operation")
        if self.operation in {"in", "not_in"} and not isinstance(self.value, (tuple, list)):
            raise ValueError("in/not_in filters require an explicit sequence of values")
        if (
            self.operation in {"greater_than", "greater_or_equal", "less_than", "less_or_equal"}
            and self.value is None
        ):
            raise ValueError("ordered row filters require a non-NULL comparison value")

    def matches(self, row: Mapping[str, Any], dtype: DType | None = None) -> bool:
        """Whether ``row`` passes this filter; the row must hold the column."""
        if self.column not in row:
            raise ValueError(f"row is missing filter column {self.column!r}")
        actual = row[self.column]
        operand = self.value
        if dtype is not None and self.operation not in {"is_null", "not_null"}:
            operand = (
                tuple(dtype.normalize(value, path=f"filter.{self.column}") for value in operand)
                if self.operation in {"in", "not_in"}
                else dtype.normalize(operand, path=f"filter.{self.column}")
            )
        if self.operation == "equal":
            return actual == operand
        if self.operation == "not_equal":
            return actual != operand
        if self.operation == "in":
            return actual in operand
        if self.operation == "not_in":
            return actual not in operand
        if self.operation == "greater_than":
            return actual is not None and actual > operand
        if self.operation == "greater_or_equal":
            return actual is not None and actual >= operand
        if self.operation == "less_than":
            return actual is not None and actual < operand
        if self.operation == "less_or_equal":
            return actual is not None and actual <= operand
        if self.operation == "is_null":
            return actual is None
        if self.operation == "not_null":
            return actual is not None
        raise ValueError(f"unsupported filter operation {self.operation!r}")


@dataclass(frozen=True)
class ActiveInterval:
    """Columns describing an active interval.

    A row is eligible when its start is strictly before the anchor and its end
    is absent or strictly after it. Both columns must be ``timestamp`` or
    ``date32``. DiskGraph refuses active intervals.
    """

    start_column: str
    end_column: str


@dataclass(frozen=True)
class TimeWindow:
    """A half-open time window relative to each sample's prediction time.

    ``start`` is exclusive and ``stop`` is inclusive. ``None`` means the bound
    is open, so ``TimeWindow("at", stop=0)`` means every row at or before
    the prediction time and ``TimeWindow("at", start=0, stop=None)`` means every
    row strictly after it.

    Bounds are offsets in seconds from the anchor. Historical context stops at
    or before zero; generated targets start at or after zero. The column, like
    every column compared with an anchor time, must be a ``timestamp`` or
    ``date32``; ``check`` refuses any other type.
    """

    column: str
    start: int | None = None
    stop: int | None = 0

    def __post_init__(self) -> None:
        if not self.column:
            raise ValueError("time-window column must be non-empty")
        if self.start is not None and self.stop is not None and self.start >= self.stop:
            raise ValueError("time window must start before it stops")


@dataclass(frozen=True)
class Rows:
    """Rows reached by following relationships from the sample entity.

    When ``time`` is omitted, the destination table's declared clock is used
    with the historical window ending at the sample time.

    :param path: relation names to follow in order; empty selects the entity row
        itself. A relation may be traversed in either direction.
    :param time: the window relative to the anchor; its column must be a
        ``timestamp`` or ``date32``.
    :param filters: typed comparisons on destination columns.
    :param active: restricts to rows whose validity interval contains the
        prediction time, which is a different question from when they occurred.
    :param distinct: after traversal, keep one visit per destination row,
        identified by its primary key or its occurrence when keyless. Separate
        identical source rows remain separate; joins can revisit the same row.
    """

    path: tuple[str, ...] = ()
    time: TimeWindow | None = None
    filters: tuple[RowFilter, ...] = ()
    active: ActiveInterval | None = None
    distinct: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(self.path))
        object.__setattr__(self, "filters", tuple(self.filters))


@dataclass(frozen=True)
class Field:
    """A typed value read from a stored column.

    :param column: source column whose dtype comes from the table schema.
    :param path: relations followed from the current row before reading; empty
        reads the current row.
    :param selection: which row to use when ``path`` returns several:
        ``"one"`` expects at most one, ``"all"`` keeps every occurrence (not a
        valid calculation operand), ``"latest"`` takes the latest eligible row
        and needs a clock, inherited or given by ``time_column``.
    :param time_column: overrides the destination table's declared clock.
    :param reference_time: whether the point-in-time lookup is as of the sample
        anchor (``"sample"``) or, for an event field, as of that event
        (``"event"``).

    The DiskGraph engine doesn't support a joined field (nonempty ``path``) yet;
    see the limitations page.
    """

    column: str
    path: tuple[str, ...] = ()
    selection: Selection = "one"
    time_column: str | None = None
    reference_time: ReferenceTime = "sample"

    def __post_init__(self) -> None:
        if not self.column:
            raise ValueError("a field requires a column name")
        if self.selection not in {"one", "all", "latest"}:
            raise ValueError(f"unsupported field selection {self.selection!r}")
        if self.reference_time not in {"sample", "event"}:
            raise ValueError(f"unsupported reference time {self.reference_time!r}")


@dataclass(frozen=True)
class Constant:
    """An explicitly typed scalar, including a typed NULL.

    ``Constant(value=None, dtype=...)`` is a typed NULL. The value is validated
    against ``dtype``, never converted.
    """

    value: Scalar | None
    dtype: DType

    def __post_init__(self) -> None:
        _result_dtype(self.dtype)
        if self.dtype is None:
            raise ValueError("constant requires an explicit dtype")
        object.__setattr__(self, "value", self.dtype.normalize(self.value, path="constant"))


def _central_moment(values: Sequence[Any], *, ddof: int) -> float | None:
    """Mean squared deviation over ``len(values) - ddof``.

    ``ddof=0`` is the population statistic, ``ddof=1`` the sample one (Bessel's
    correction). The sample form needs at least two observations because
    spread around an estimated mean is undefined from a single point; the
    contract represents that result as NULL, distinct from a floating NaN.
    """

    degrees = len(values) - ddof
    if degrees <= 0:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / degrees


def _validate_reduction(operation: SummaryOperation, column: str | None) -> None:
    if operation not in ALL_REDUCTIONS:
        raise ValueError(f"unsupported reduction operation {operation!r}")
    if operation != "count" and column is None:
        raise ValueError(f"{operation} requires a column")


@dataclass(frozen=True)
class Reduce:
    """One aggregation performed inside each group of a :class:`GroupBy`.

    Takes the same operations, result types and missing-value rules as
    :class:`Aggregate`.

    :param column: the column reduced; omitted only for ``count``.
    :param dtype: optional checked result type.
    """

    operation: SummaryOperation
    column: str | None = None
    dtype: DType | None = dataclass_field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _result_dtype(self.dtype)
        _validate_reduction(self.operation, self.column)


@dataclass(frozen=True)
class GroupBy:
    """Turn rows into one row per key with explicitly named aggregate values.

    A grouped result can be the source of another ``GroupBy`` or ``Aggregate``.
    This is enough to express, for example, the mean of per-account transaction
    counts without adding a model-specific feature class.

    :param source: rows to group, or another grouping to group again.
    :param keys: columns whose distinct combinations define one group.
    :param values: output name to the reduction producing it.
    """

    source: Rows | GroupBy
    keys: tuple[str, ...]
    values: Mapping[str, Reduce]

    def __post_init__(self) -> None:
        if not self.keys or len(self.keys) != len(set(self.keys)):
            raise ValueError("group-by keys must be non-empty and unique")
        if not self.values or any(not name for name in self.values):
            raise ValueError("group-by values must have non-empty names")
        overlap = set(self.keys) & set(self.values)
        if overlap:
            raise ValueError(f"group-by keys and values overlap: {sorted(overlap)}")
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "values", _mapping(self.values))


RowSource: TypeAlias = Rows | GroupBy


@dataclass(frozen=True)
class Aggregate:
    """Reduce a row source to one typed value.

    The same object is valid in ``SampleSpec.entity_values`` and
    ``SampleSpec.targets``. Its placement determines whether it is context or
    supervision; the time window determines which rows are eligible.

    Operations and result types: ``count`` and ``count_distinct`` give
    ``int64``; ``sum`` gives exact ``int64`` for signed and ``uint64`` for
    unsigned integers (a range error, never wraparound), ``float64`` for floats
    (native rounding and overflow kept); ``mean``, ``std_population``,
    ``std_sample``, ``var_population`` and ``var_sample`` give ``float64``;
    ``min`` and ``max`` (numerical, string, date, timestamp) and ``mode`` (any
    scalar) keep the column's type; ``recency`` gives ``float64`` seconds.

    NULL values are ignored. An empty row set gives zero for ``count``,
    ``count_distinct`` and ``sum``; a nonempty set whose values are all NULL
    gives a missing sum; other reductions without observations are missing.
    Population spreads divide by ``n`` and sample spreads by ``n - 1``, so one
    observation gives population spread zero and a missing sample spread.
    ``mode`` breaks frequency ties by ``(type name, repr(value))``. Raw float
    ``count_distinct`` and ``mode`` compare IEEE bit patterns, so positive and
    negative zero differ. DiskGraph refuses reductions over bytes and decimals.

    :param source: the rows (or grouped rows) reduced.
    :param operation: the reduction.
    :param column: the column reduced; omitted only for ``count``.
    :param dtype: optional checked result type; omitted types follow the operation.
    """

    source: RowSource
    operation: SummaryOperation
    column: str | None = None
    dtype: DType | None = dataclass_field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _result_dtype(self.dtype)
        _validate_reduction(self.operation, self.column)


@dataclass(frozen=True)
class Compare:
    """Compare an aggregate result with one compatible scalar value.

    The result is a typed boolean; a missing aggregate stays missing.
    """

    left: Aggregate
    operation: ComparisonOperation
    right: Scalar
    dtype: DType = dataclass_field(default=DType("bool"), kw_only=True)

    def __post_init__(self) -> None:
        _result_dtype(self.dtype)
        if self.operation not in _ALL_COMPARISON_OPERATIONS:
            raise ValueError(f"unsupported comparison operation {self.operation!r}")
        if self.dtype != DType("bool"):
            raise ValueError("comparison result dtype must be bool")
        if self.right is None:
            raise ValueError("comparison literals must be non-NULL")
        if self.left.dtype is not None:
            object.__setattr__(
                self, "right", self.left.dtype.normalize(self.right, path="comparison.right")
            )
        if (
            self.left.dtype is not None
            and self.operation not in {"equal", "not_equal"}
            and not (
                self.left.dtype.numeric or self.left.dtype.name in {"string", "date32", "timestamp"}
            )
        ):
            raise ValueError(f"{self.left.dtype.name} only supports equal/not_equal comparison")


@dataclass(frozen=True)
class CalendarPart:
    """Extract one UTC calendar component from a timestamp or date.

    Use it as ``Derived(operation=CalendarPart(...), inputs=(name,))``. The
    result is ``int64``. ``date32`` inputs accept date components only.
    """

    part: Literal["year", "month", "day", "weekday", "hour", "minute", "second"]
    timezone: Literal["UTC"] = "UTC"

    def __post_init__(self) -> None:
        if self.part not in {"year", "month", "day", "weekday", "hour", "minute", "second"}:
            raise ValueError(f"unsupported calendar part {self.part!r}")
        if self.timezone != "UTC":
            raise ValueError("calendar extraction supports timezone='UTC'")


@dataclass(frozen=True)
class MapValues:
    """Map disjoint sets of source values to explicitly supplied result values.

    Used as a :class:`Derived` operation, which must declare the result
    ``dtype``. Members are normalized to the operand type before overlap is
    checked, and must be finite and non-NULL. A NULL operand stays NULL rather
    than selecting a group or the fallback.

    :param groups: result value to the source values that map to it.
    :param unknown: policy for a value in no group: ``"error"``, ``"null"``, or
        ``"fallback"`` with a typed ``fallback``.
    """

    groups: Mapping[Scalar, tuple[Scalar, ...]]
    unknown: Literal["error", "null", "fallback"] = "error"
    fallback: Scalar | None = None

    def __post_init__(self) -> None:
        if self.unknown not in {"error", "null", "fallback"}:
            raise ValueError("map unknown policy must be error, null, or fallback")
        if self.unknown != "fallback" and self.fallback is not None:
            raise ValueError("map fallback requires unknown='fallback'")
        members = [(type(member), member) for values in self.groups.values() for member in values]
        if len(members) != len(set(members)):
            raise ValueError("map groups must have disjoint members")
        object.__setattr__(
            self,
            "groups",
            FrozenMapping({key: tuple(values) for key, values in self.groups.items()}),
        )


@dataclass(frozen=True)
class Derived:
    """Arithmetic over named values in the same entity or candidate group.

    Operations: ``add``, ``subtract``, ``multiply`` (Arrow numeric promotion,
    for example int64 plus float64 gives float64; decimal multiplication adds
    the operand scales); ``divide`` and ``log1p`` (``float64``); ``abs`` (same
    family); ``sign`` (``int64`` -1, 0 or 1); and the configured
    :class:`CalendarPart` and :class:`MapValues`. Numerical operands exclude
    booleans. A missing operand gives a missing result; division by zero and
    ``log1p(x)`` for ``x <= -1`` give missing values. Integer and decimal results
    are checked against their type's range; floating results keep native
    rounding and overflow, including infinity.

    :param operation: the fixed operation; not a user function.
    :param inputs: value names, not columns. Prerequisites are evaluated first,
        regardless of declaration placement. Operand order is preserved.
        Operands must be scalar.
    :param dtype: optional checked expectation, or a compatible widening:
        a wider integer, float32 to float64, or a larger decimal precision.
    """

    operation: DerivedOperation | CalendarPart | MapValues
    inputs: tuple[str, ...]
    dtype: DType | None = dataclass_field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _result_dtype(self.dtype)
        if isinstance(self.operation, MapValues) and self.dtype is None:
            raise ValueError("MapValues requires an explicit result dtype")
        expected = {
            "add": 2,
            "subtract": 2,
            "multiply": 2,
            "divide": 2,
            "log1p": 1,
            "abs": 1,
            "sign": 1,
        }
        arity = (
            1
            if isinstance(self.operation, (CalendarPart, MapValues))
            else expected.get(self.operation)
        )
        if arity is None:
            raise ValueError(f"unsupported derived operation {self.operation!r}")
        if len(self.inputs) != arity:
            raise ValueError(f"{self.operation} requires {arity} input field(s)")
        object.__setattr__(self, "inputs", tuple(self.inputs))


EntityValue: TypeAlias = Field | Constant | Aggregate | Compare | Derived


def _value_evaluation_order(values: Mapping[str, EntityValue]) -> tuple[str, ...]:
    """Resolve named prerequisites without assigning model-input positions."""

    dependencies = {}
    for name in sorted(values):
        value = values[name]
        inputs = set(value.inputs) if isinstance(value, Derived) else set()
        unknown = inputs - values.keys()
        if unknown:
            raise ValueError(f"value {name!r} references unknown value(s): {sorted(unknown)}")
        dependencies[name] = sorted(inputs)
    try:
        return tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as error:
        raise ValueError(f"cyclic value dependencies: {' -> '.join(error.args[1])}") from error


@dataclass(frozen=True)
class DatedEntityValue:
    """One repeated, timestamped value associated with the entity.

    :param rows: the timed rows holding the occurrences.
    :param field: the value of each occurrence.
    :param first_per_value: keep only the earliest occurrence of each distinct
        value, turning a history into "when did this first become true".
    :param fields: additional typed values per retained occurrence, such as an
        ``EventTimeDelta`` age; ``OccurrencesOf`` projects them onto token
        positions.

    The DiskGraph engine doesn't support this yet; see the limitations page.
    """

    rows: Rows
    field: Field | Constant
    first_per_value: bool = False
    fields: Mapping[str, Field | Constant | Derived | EventTimeDelta] = dataclass_field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if any(not name for name in self.fields):
            raise ValueError("dated occurrence fields require non-empty names")
        object.__setattr__(self, "fields", _mapping(self.fields))


@dataclass(frozen=True)
class EventTimeDelta:
    """Elapsed time from an event to its sample anchor, emitted as a number.

    The value is ``(anchor time - event time)`` in ``unit``; it is not relative
    to the latest event. Nothing like it is emitted unless declared.

    :param unit: unit of the emitted number. Row-window offsets stay in seconds
        regardless.
    """

    unit: Literal["seconds", "minutes", "hours", "days"] = "seconds"
    dtype: DType = dataclass_field(default=DType("float64"), kw_only=True)

    def __post_init__(self) -> None:
        _result_dtype(self.dtype)
        if self.unit not in {"seconds", "minutes", "hours", "days"}:
            raise ValueError("event time-delta unit must be seconds, minutes, hours, or days")
        if self.dtype != DType("float64"):
            raise ValueError("event time-delta result dtype must be float64")

    @property
    def seconds_per_unit(self) -> int:
        """Seconds in one ``unit``."""
        return {"seconds": 1, "minutes": 60, "hours": 3_600, "days": 86_400}[self.unit]


@dataclass(frozen=True)
class EventSpec:
    """Rows and fields for one named chronological event component.

    Fields can be columns, constants, calculations over the event's own fields,
    or an explicit ``EventTimeDelta``. Timestamps, calendar parts and ages are
    emitted only when declared as fields.

    :param rows: the events' rows and window.
    :param fields: named values per event.
    :param limit: maximum events kept, applied *after* ``order``, so
        ``order="descending"`` keeps the most recent and ``"ascending"`` the
        oldest.
    :param order: chronological direction the events are sorted in.
    """

    rows: Rows
    fields: Mapping[str, Field | Constant | Derived | EventTimeDelta]
    limit: int | None = None
    order: Literal["ascending", "descending"] = "ascending"

    def __post_init__(self) -> None:
        if not self.fields or any(not name for name in self.fields):
            raise ValueError("events require at least one field")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("event limit must be positive")
        if self.order not in {"ascending", "descending"}:
            raise ValueError("event order must be 'ascending' or 'descending'")
        object.__setattr__(self, "fields", _mapping(self.fields))


@dataclass(frozen=True)
class RowTarget:
    """Select one future row and expose configured fields as one target event.

    :param rows: future rows; the window's lower bound is at or after zero and
        excluded.
    :param fields: values exposed from the selected row.
    :param selection: which row of the window to take, in chronological order.
    """

    rows: Rows
    fields: Mapping[str, Field | Constant | Derived | EventTimeDelta]
    selection: Literal["first", "last"] = "first"

    def __post_init__(self) -> None:
        if self.rows.time is None:
            raise ValueError("row targets require a time window")
        if not self.fields or any(not name for name in self.fields):
            raise ValueError("row targets require at least one field")
        if self.selection not in {"first", "last"}:
            raise ValueError("row target selection must be 'first' or 'last'")
        object.__setattr__(self, "fields", _mapping(self.fields))


@dataclass(frozen=True)
class GraphSample:
    """One bounded relationship traversal used to select neighboring rows.

    :param fanout: maximum neighbors kept per frontier row.
    :param direction: which way to follow the relation; ``"forward"`` is left
        table to right as written in :class:`RelationSpec`.
    :param relation: the declared relation followed.
    :param strategy: how to choose when more neighbors exist than ``fanout``:
        ``"recent"`` (the default) keeps the most recent and needs a clock, even
        when ``temporal="static"``; ``"uniform"`` draws evenly among the candidates. Which tied
        neighbors at the fanout limit are kept is unspecified (G-GRAPH-001). The
        DiskGraph engine supports ``"recent"`` only.

    Emitted edges keep the relation's left-to-right orientation whatever the
    traversal direction, self-relations included.
    :param time_column: overrides the destination table's declared clock.
    :param temporal: ``"inherit"`` filters by the resolved clock; ``"static"``
        states that this traversal is deliberately unfiltered.
    """

    relation: str
    fanout: int
    direction: Literal["forward", "reverse"] = "forward"
    strategy: Literal["recent", "uniform"] = "recent"
    time_column: str | None = None
    temporal: Literal["inherit", "static"] = "inherit"

    def __post_init__(self) -> None:
        if self.fanout <= 0:
            raise ValueError("graph fanout must be positive")
        if self.direction not in {"forward", "reverse"}:
            raise ValueError("graph direction must be 'forward' or 'reverse'")
        if self.strategy not in {"uniform", "recent"}:
            raise ValueError("graph strategy must be 'uniform' or 'recent'")
        if self.temporal not in {"inherit", "static"}:
            raise ValueError("graph temporal must be 'inherit' or 'static'")


@dataclass(frozen=True)
class GraphLayer:
    """Relationship traversals performed from one neighborhood frontier."""

    samples: tuple[GraphSample, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("graph layer requires at least one relation sample")


@dataclass(frozen=True)
class NeighborhoodSpec:
    """Rows belonging to one sampled neighborhood around the entity.

    Layers expand in order from the entity. A row or edge reached more than once
    appears once per sample. Node attributes are chosen in ``BatchSpec`` through
    a ``Graph`` representation, not here.
    """

    layers: tuple[GraphLayer, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("neighborhood requires at least one layer")


@dataclass(frozen=True)
class EntitySpec:
    """The table each sample is about, identified by its schema primary key.

    Requests identify entities by the table's primary key; there is no second
    lookup column.

    :param table: the entity table; its schema primary key supplies identity.
    :param entity_type: the name this entity carries downstream, e.g. the node
        type in a graph batch; defaults to ``table``.
    :param filters: eligibility conditions on the entity row.
    :param available_from: column holding when the entity came into existence.
        A request anchored at or before it is refused rather than answered.
    """

    table: str
    entity_type: str | None = dataclass_field(default=None, kw_only=True)
    filters: tuple[RowFilter, ...] = dataclass_field(default=(), kw_only=True)
    available_from: str | None = dataclass_field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.entity_type is None:
            object.__setattr__(self, "entity_type", self.table)


@dataclass(frozen=True)
class ExternalTarget:
    """The type of a caller-supplied label; an absent label remains missing.

    Labels arrive on ``SampleRequest.targets`` or an anchor-table column. A
    supplied label cannot replace a generated target of the same name, and an
    external label alone does not make a contract temporal.
    """

    dtype: DType

    def __post_init__(self) -> None:
        _result_dtype(self.dtype)
        if self.dtype is None:
            raise ValueError("external target requires an explicit dtype")


@dataclass(frozen=True)
class CandidateSetSpec:
    """Point-in-time values and optional generated targets for named candidates.

    Each candidate uses the primary request's time, and its values and targets
    follow the same typed sampling rules as the entity's. Candidate generation
    is the caller's. The DiskGraph engine doesn't support candidate sets yet;
    see the limitations page.

    :param entity: what a candidate *is*; candidates are supplied per request,
        not selected here.
    """

    entity: EntitySpec
    values: Mapping[str, EntityValue] = dataclass_field(default_factory=dict)
    targets: Mapping[str, Aggregate | Compare | RowTarget | ExternalTarget] = dataclass_field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if any(not str(name) for name in self.values):
            raise ValueError("candidate value names must be non-empty")
        if any(not str(name) for name in self.targets):
            raise ValueError("candidate target names must be non-empty")
        object.__setattr__(self, "values", _mapping(self.values))
        object.__setattr__(self, "targets", _mapping(self.targets))

    def sample_spec(self) -> SampleSpec:
        """Use normal sample construction for each caller-supplied candidate."""

        return SampleSpec(
            entity=self.entity,
            entity_values=self.values,
            targets=self.targets,
        )


@dataclass(frozen=True)
class SampleSpec:
    """Named raw data components belonging to one entity at one point in time.

    Values here are typed raw data, not encodings: how they reach a model is
    chosen in ``BatchSpec``. Named calculations may refer to each other in any
    declaration order; unknown names and cycles are refused. A contract needs an
    anchor time when a table it reads has a clock or a value depends on time;
    see ``DataLoaderSpec.requires_anchor_time``.

    :param entity: the table each sample is about.
    :param entity_values: named scalar values: fields, constants, aggregates,
        comparisons and calculations.
    :param events: named historical event components. Several components can
        merge into one batch input; equal-time events then order by component
        name and primary key, or local occurrence identity for a keyless table.
    :param dated_entity_values: repeated timestamped facts, as opposed to the
        single values in ``entity_values``.
    :param current_events: events kept apart from ``events`` so an objective can
        distinguish the event being predicted from its history. Each requires
        ``TimeWindow(column=..., start=-1, stop=0)`` and ``limit=1``, selecting
        the event at the anchor's stored second; a historical window ending at
        zero still includes that event too.
    :param neighborhoods: named graph neighborhoods around the entity.
    :param targets: labels computed from future rows. Labels already known to
        the caller arrive on the request instead and are declared here with
        ``ExternalTarget``. A computed target needs an explicit future window
        whose lower bound is at or after zero; that bound is excluded, so a
        target never reads an event at the anchor itself. Declaring a target
        never inserts it into an input.
    :param candidate_sets: how to build the candidates a request supplies.
    """

    entity: EntitySpec
    entity_values: Mapping[str, EntityValue] = dataclass_field(default_factory=dict)
    dated_entity_values: Mapping[str, DatedEntityValue] = dataclass_field(default_factory=dict)
    events: Mapping[str, EventSpec] = dataclass_field(default_factory=dict)
    current_events: Mapping[str, EventSpec] = dataclass_field(default_factory=dict)
    neighborhoods: Mapping[str, NeighborhoodSpec] = dataclass_field(default_factory=dict)
    targets: Mapping[str, Aggregate | Compare | RowTarget | ExternalTarget] = dataclass_field(
        default_factory=dict
    )
    candidate_sets: Mapping[str, CandidateSetSpec] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, values in (
            ("entity value", self.entity_values),
            ("dated entity value", self.dated_entity_values),
            ("event", self.events),
            ("current event", self.current_events),
            ("neighborhood", self.neighborhoods),
            ("target", self.targets),
            ("candidate set", self.candidate_sets),
        ):
            if any(not str(name) for name in values):
                raise ValueError(f"{label} component names must be non-empty")
        overlap = set(self.entity_values) & set(self.dated_entity_values)
        if overlap:
            raise ValueError(f"entity value names must be unique: {sorted(overlap)}")
        object.__setattr__(self, "entity_values", _mapping(self.entity_values))
        object.__setattr__(self, "dated_entity_values", _mapping(self.dated_entity_values))
        object.__setattr__(self, "events", _mapping(self.events))
        object.__setattr__(self, "current_events", _mapping(self.current_events))
        object.__setattr__(self, "neighborhoods", _mapping(self.neighborhoods))
        object.__setattr__(self, "targets", _mapping(self.targets))
        object.__setattr__(self, "candidate_sets", _mapping(self.candidate_sets))


@dataclass(frozen=True)
class TargetValue:
    """A target carried beside a sample and never inserted into its inputs.

    :param name: an ``ExternalTarget`` declared in ``SampleSpec.targets``.
    :param value: the label, validated against the declared dtype.
    """

    name: str
    value: Any


def _normalize_supplied_targets(
    sample: SampleSpec, targets: Sequence[TargetValue]
) -> tuple[TargetValue, ...]:
    """Validate caller labels through the same typed rule for every executor."""
    result = []
    for target in targets:
        declaration = sample.targets.get(target.name)
        if declaration is None:
            raise ValueError(f"unknown supplied target: {target.name!r}")
        if not isinstance(declaration, ExternalTarget):
            raise ValueError(f"target {target.name!r} was both supplied and generated")
        result.append(
            TargetValue(
                target.name,
                declaration.dtype.normalize(target.value, path=f"targets.{target.name}"),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class CandidateRequest:
    """One caller-chosen candidate and any label already known for it.

    :param entity_id: primary-key value in the candidate's declared table.
    :param targets: labels supplied by the caller, for candidates whose
        supervision is not derived from rows.
    """

    entity_id: int
    targets: tuple[TargetValue, ...] = ()

    def __post_init__(self) -> None:
        names = [target.name for target in self.targets]
        if len(names) != len(set(names)):
            raise ValueError("target names must be unique within one candidate")


@dataclass(frozen=True)
class SampleRequest:
    """One requested entity/time pair and any already-derived target values.

    :param entity_id: primary-key value in the sample's declared entity table.
    :param sample_id: identifies the sample in batch metadata.
    :param as_of_time: the prediction time every window in the sample is
        relative to. Naive dates and datetimes are read as UTC. Required when
        the contract is temporal, refused when it is not.
    :param targets: labels supplied by the caller rather than computed from rows.
    :param split: free-form partition label carried through to the batch.
    :param metadata: opaque values carried through untouched.
    :param candidates: candidate-set name to the candidates to score for it.
    """

    sample_id: str | int
    entity_id: int
    as_of_time: int | dt.date | dt.datetime
    targets: tuple[TargetValue, ...] = ()
    split: str | None = None
    metadata: Mapping[str, Any] = dataclass_field(default_factory=dict)
    candidates: Mapping[str, tuple[CandidateRequest, ...]] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [target.name for target in self.targets]
        if len(names) != len(set(names)):
            raise ValueError("target names must be unique within one sample")
        if self.split is not None and not self.split:
            raise ValueError("split must be non-empty when supplied")
        object.__setattr__(self, "metadata", _mapping(self.metadata))
        object.__setattr__(
            self,
            "candidates",
            FrozenMapping(
                {name: tuple(candidates) for name, candidates in self.candidates.items()}
            ),
        )


@dataclass(frozen=True)
class BuiltSample:
    """The raw history consumed by model tokenizers plus separate targets.

    This is the engine boundary: whatever produced it, everything downstream
    sees only this.

    :param candidates: one nested sample per candidate, built the same way.
    """

    sample_id: str | int
    history: EntityHistory
    targets: tuple[TargetValue, ...]
    current_events: Mapping[str, tuple[Event, ...]] = dataclass_field(default_factory=dict)
    metadata: Mapping[str, Any] = dataclass_field(default_factory=dict)
    candidates: Mapping[str, tuple[BuiltSample, ...]] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_events",
            FrozenMapping({name: tuple(events) for name, events in self.current_events.items()}),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata))
        object.__setattr__(
            self,
            "candidates",
            FrozenMapping(
                {name: tuple(candidates) for name, candidates in self.candidates.items()}
            ),
        )


def destination_table(schema: DataSchema, start: str, path: tuple[str, ...]) -> str:
    """Resolve a relation path, rejecting a disconnected or unknown step."""

    relations = {relation.name: relation for relation in schema.relations}
    for name in path:
        relation = relations.get(name)
        if relation is None:
            raise ValueError(f"unknown relation {name!r}")
        if relation.left_table == start:
            start = relation.right_table
        elif relation.right_table == start:
            start = relation.left_table
        else:
            raise ValueError(f"relation {name!r} cannot be followed from {start!r}")
    return start


def source_column_types(schema: DataSchema, start: str, source: RowSource) -> Mapping[str, DType]:
    if isinstance(source, Rows):
        table = destination_table(schema, start, source.path)
        return next(item.columns for item in schema.tables if item.name == table)
    columns = source_column_types(schema, start, source.source)
    result = {name: columns[name] for name in source.keys}
    for name, reduction in source.values.items():
        result[name] = reduction_dtype(reduction, columns)
    return result


def _can_widen(source: DType, result: DType) -> bool:
    if source == result:
        return True
    if source.integer and result.integer:
        return source.name.startswith("u") == result.name.startswith("u") and int(
            result.name.lstrip("uint")
        ) >= int(source.name.lstrip("uint"))
    if source.name == "float32" and result.name == "float64":
        return True
    return (
        source.name.startswith("decimal")
        and result.name.startswith("decimal")
        and source.scale == result.scale
        and result.precision >= source.precision
    )


def reduction_dtype(spec: Reduce | Aggregate, columns: Mapping[str, DType]) -> DType:
    """The native scalar reduction result, checked against an optional declaration."""
    operation = spec.operation
    source = None if spec.column is None else columns[spec.column]
    if operation in {"count", "count_distinct"}:
        inferred = DType("int64")
    elif operation == "sum":
        if source.integer:
            inferred = DType("int64" if source.name.startswith("int") else "uint64")
        elif source.name.startswith("decimal"):
            inferred = source
        else:
            inferred = DType("float64")
    elif operation in {
        "mean",
        "std_population",
        "std_sample",
        "var_population",
        "var_sample",
        "recency",
    }:
        inferred = DType("float64")
    else:
        inferred = source
    result = spec.dtype or inferred
    if operation in {"count", "count_distinct"}:
        valid = result == DType("int64")
    elif operation in {"mean", "std_population", "std_sample", "var_population", "var_sample"}:
        valid = source.numeric and result == DType("float64")
    elif operation == "recency":
        valid = source.temporal and result == DType("float64")
    elif operation == "sum":
        valid = source.numeric and (
            _can_widen(source, result) if source.name.startswith("decimal") else result == inferred
        )
    elif operation in {"min", "max"}:
        valid = (
            source.numeric or source.name in {"string", "date32", "timestamp"}
        ) and result == source
    else:
        valid = result == source
    if not valid:
        raise ValueError(f"{operation} cannot produce {result} from {source}")
    return result


def _arithmetic_dtype(operation: str, operands: Sequence[DType]) -> DType:
    """Use the same Arrow numeric promotion as columnar execution."""
    import pyarrow as pa
    import pyarrow.compute as pc

    function = "abs_checked" if operation == "abs" else operation + "_checked"
    try:
        result = getattr(pc, function)(*(pa.array([], type=item.to_arrow()) for item in operands))
    except pa.ArrowException as error:
        raise ValueError(f"derived {operation} cannot combine {operands}: {error}") from error
    return DType.from_arrow(result.type)


def value_dtype(
    schema: DataSchema,
    root_table: str,
    declaration: EntityValue | EventTimeDelta,
    values: Mapping[str, EntityValue | EventTimeDelta] | None = None,
) -> DType:
    """Resolve and check a sample value's dtype without reading source rows."""

    if isinstance(declaration, Field):
        table = destination_table(schema, root_table, declaration.path)
        tables = {item.name: item for item in schema.tables}
        return tables[table].columns[declaration.column]
    if isinstance(declaration, Aggregate):
        return reduction_dtype(
            declaration, source_column_types(schema, root_table, declaration.source)
        )
    if isinstance(declaration, Compare):
        value_dtype(schema, root_table, declaration.left)
    if isinstance(declaration, Derived):
        if values is None:
            raise ValueError("derived values require their named declaration scope")
        operands = []
        for name in declaration.inputs:
            source = values[name]
            if isinstance(source, Field) and source.selection == "all":
                raise ValueError(
                    "derived calculations require scalar operands, not selection='all'"
                )
            operands.append(value_dtype(schema, root_table, source, values))
        operation = declaration.operation
        if isinstance(operation, MapValues):
            inferred = declaration.dtype
        elif isinstance(operation, CalendarPart) or operation == "sign":
            inferred = DType("int64")
        elif operation in {"divide", "log1p"}:
            inferred = DType("float64")
        elif not all(item.numeric for item in operands):
            inferred = None
        elif (
            declaration.dtype is not None
            and all(item == operands[0] for item in operands)
            and operands[0].name.startswith("decimal")
        ):
            inferred = operands[0]
        else:
            inferred = _arithmetic_dtype(operation, operands)
        result = declaration.dtype or inferred
        if isinstance(operation, CalendarPart):
            valid = operands[0].temporal and result == DType("int64")
            if operands[0].name == "date32" and operation.part in {"hour", "minute", "second"}:
                valid = False
        elif isinstance(operation, MapValues):
            for output, members in _typed_map_groups(operation, operands[0]):
                result.normalize(output, path="map result")
            if operation.unknown == "fallback":
                result.normalize(operation.fallback, path="map fallback")
            valid = True
        elif operation in {"divide", "log1p"}:
            valid = all(item.numeric for item in operands) and result == DType("float64")
        elif operation == "sign":
            valid = operands[0].numeric and result == DType("int64")
        else:
            valid = all(item.numeric for item in operands) and _can_widen(inferred, result)
            if (
                all(item == operands[0] for item in operands)
                and operands[0].name.startswith("decimal")
                and declaration.dtype is not None
            ):
                source = operands[0]
                valid = (
                    result.name.startswith("decimal")
                    and result.scale
                    == (2 * source.scale if operation == "multiply" else source.scale)
                    and result.precision >= source.precision
                )
        if not valid:
            raise ValueError(f"derived {operation!r} cannot produce {result} from {operands}")
        return result
    return declaration.dtype


def resolve_sample_types(schema: DataSchema, sample: SampleSpec) -> SampleSpec:
    """Return concrete calculation types for every sampled scope."""

    def unresolved(value: Any) -> bool:
        if isinstance(value, (Aggregate, Reduce, Derived)) and value.dtype is None:
            return True
        if is_dataclass(value):
            return any(unresolved(getattr(value, item.name)) for item in fields(value))
        if isinstance(value, Mapping):
            return any(unresolved(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return any(unresolved(item) for item in value)
        return False

    if not unresolved(sample):
        return sample

    def source(value: RowSource, table: str) -> RowSource:
        if isinstance(value, Rows):
            return value
        parent = source(value.source, table)
        columns = source_column_types(schema, table, parent)
        return replace(
            value,
            source=parent,
            values={
                name: replace(item, dtype=item.dtype or reduction_dtype(item, columns))
                for name, item in value.values.items()
            },
        )

    def scope(values: Mapping[str, Any], table: str) -> dict[str, Any]:
        if table not in {item.name for item in schema.tables}:
            raise ValueError(f"sample scope references unknown table {table!r}")
        result = dict(values)
        for name in _value_evaluation_order(values):
            item = values[name]
            if isinstance(item, RowTarget):
                result[name] = replace(
                    item,
                    fields=scope(item.fields, destination_table(schema, table, item.rows.path)),
                )
            elif isinstance(item, Compare):
                result[name] = replace(item, left=scope({"left": item.left}, table)["left"])
            elif isinstance(item, Aggregate):
                item = replace(item, source=source(item.source, table))
                result[name] = replace(
                    item, dtype=item.dtype or value_dtype(schema, table, item, result)
                )
            elif isinstance(item, Derived) and item.dtype is None:
                result[name] = replace(item, dtype=value_dtype(schema, table, item, result))
        return result

    def events(values: Mapping[str, EventSpec]) -> dict[str, EventSpec]:
        return {
            name: replace(
                item, fields=scope(item.fields, destination_table(schema, root, item.rows.path))
            )
            for name, item in values.items()
        }

    root = sample.entity.table
    dated = {}
    for name, item in sample.dated_entity_values.items():
        if name in item.fields:
            raise ValueError(f"dated occurrence field {name!r} duplicates its primary value")
        values = scope(
            {name: item.field, **item.fields}, destination_table(schema, root, item.rows.path)
        )
        dated[name] = replace(item, field=values.pop(name), fields=values)
    return replace(
        sample,
        entity_values=scope(sample.entity_values, root),
        dated_entity_values=dated,
        events=events(sample.events),
        current_events=events(sample.current_events),
        targets=scope(sample.targets, root),
        candidate_sets={
            name: replace(
                item,
                values=scope(item.values, item.entity.table),
                targets=scope(item.targets, item.entity.table),
            )
            for name, item in sample.candidate_sets.items()
        },
    )


def _scope_types(schema: DataSchema, table: str, values: Mapping[str, Any]) -> dict[str, DType]:
    order = _value_evaluation_order(values)
    return {name: value_dtype(schema, table, values[name], values) for name in order}


def sample_value_types(schema: DataSchema, sample: SampleSpec) -> dict[str, DType]:
    result = _scope_types(schema, sample.entity.table, sample.entity_values)
    for name, dated in sample.dated_entity_values.items():
        table = destination_table(schema, sample.entity.table, dated.rows.path)
        result[name] = value_dtype(schema, table, dated.field)
    return result


def dated_value_types(schema: DataSchema, sample: SampleSpec) -> dict[str, dict[str, DType]]:
    result = {}
    for name, dated in sample.dated_entity_values.items():
        if name in dated.fields:
            raise ValueError(f"dated occurrence field {name!r} duplicates its primary value")
        table = destination_table(schema, sample.entity.table, dated.rows.path)
        result[name] = _scope_types(schema, table, {name: dated.field, **dated.fields})
    return result


def event_value_types(
    schema: DataSchema, sample: SampleSpec, *, current: bool = False
) -> dict[str, dict[str, DType]]:
    return {
        name: _scope_types(
            schema, destination_table(schema, sample.entity.table, event.rows.path), event.fields
        )
        for name, event in (sample.current_events if current else sample.events).items()
    }


def _typed_map_groups(
    operation: MapValues, dtype: DType
) -> tuple[tuple[Any, tuple[Any, ...]], ...]:
    """Resolve literal membership in the operand's actual stored scalar domain."""
    groups, seen = [], set()
    for output, members in operation.groups.items():
        normalized = tuple(dtype.normalize(member, path="map member") for member in members)
        for member in normalized:
            if member is None:
                raise ValueError("map members must be non-NULL; NULL operands remain NULL")
            if isinstance(member, float) and not math.isfinite(member):
                raise ValueError("map members require finite scalar identities")
            key = (type(member), member)
            if key in seen:
                raise ValueError("map groups overlap after dtype normalization")
            seen.add(key)
        groups.append((output, normalized))
    return tuple(groups)


def evaluate_derived(
    spec: Derived,
    values: Mapping[str, Any],
    *,
    path: str,
    input_types: Mapping[str, DType] | None = None,
) -> Any:
    """Evaluate one checked fixed calculation in its named scalar scope."""

    operands = [values.get(name) for name in spec.inputs]
    if any(value is None for value in operands):
        return None
    operation = spec.operation
    if isinstance(operation, CalendarPart):
        value = operands[0]
        if isinstance(value, dt.datetime):
            value = (
                value.replace(tzinfo=dt.UTC) if value.tzinfo is None else value.astimezone(dt.UTC)
            )
        result = value.weekday() if operation.part == "weekday" else getattr(value, operation.part)
    elif isinstance(operation, MapValues):
        if input_types is None:
            raise ValueError(f"{path}: MapValues requires the resolved operand dtype")
        for output, members in _typed_map_groups(operation, input_types[spec.inputs[0]]):
            if any(
                type(operands[0]) is type(member) and operands[0] == member for member in members
            ):
                result = output
                break
        else:
            if operation.unknown == "error":
                raise ValueError(f"{path}: unmapped value {operands[0]!r}")
            result = operation.fallback if operation.unknown == "fallback" else None
    elif operation in {"add", "subtract", "multiply"}:
        if spec.dtype.name.startswith("float"):
            operands = [spec.dtype.normalize(float(value), path=path) for value in operands]
        elif spec.dtype.integer:
            operands = [spec.dtype.normalize(value, path=path) for value in operands]
        with localcontext() as context:
            context.prec = 160
            if operation == "add":
                result = operands[0] + operands[1]
            elif operation == "subtract":
                result = operands[0] - operands[1]
            else:
                result = operands[0] * operands[1]
    elif operation == "divide":
        result = None if operands[1] == 0 else float(operands[0]) / float(operands[1])
    elif operation == "log1p":
        result = None if operands[0] <= -1 else math.log1p(float(operands[0]))
    elif operation == "abs":
        result = operands[0].copy_abs() if isinstance(operands[0], Decimal) else abs(operands[0])
    else:
        value = operands[0]
        result = (
            value
            if isinstance(value, float) and math.isnan(value)
            else int(value > 0) - int(value < 0)
        )
    if spec.dtype.name == "float32" and result is not None:
        import pyarrow as pa

        result = pa.scalar(float(result)).cast(pa.float32(), safe=False).as_py()
    return spec.dtype.normalize(result, path=path)
