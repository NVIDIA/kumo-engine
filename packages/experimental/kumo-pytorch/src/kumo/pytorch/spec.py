# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical typed contracts, validation, and serialization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any

from kumo.pytorch.contract._frozen import FrozenMapping
from kumo.pytorch.contract.batch import BatchSpec, Fields, Graph, UseEncoding
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import (
    ActiveInterval,
    Aggregate,
    Compare,
    Constant,
    Derived,
    EventTimeDelta,
    ExternalTarget,
    Field,
    GroupBy,
    RowFilter,
    Rows,
    RowTarget,
    SampleSpec,
    TimeWindow,
    _value_evaluation_order,
    destination_table,
    reduction_dtype,
    resolve_sample_types,
    source_column_types,
    value_dtype,
)
from kumo.pytorch.contract.schema import DataSchema, table_clock
from kumo.pytorch.dataset import Dataset


@dataclass(frozen=True)
class ValidationIssue:
    """A contract path, its problem, and an actionable way forward.

    :param path: where in the contract the problem is, e.g. ``batch.inputs.history``.
    :param message: what is wrong.
    :param suggestion: what to do instead, when there is an alternative.
    :param code: the stable code of an engine's declared unsupported feature, such as
        an entry of `DiskGraphEngine.unsupported`; ``None`` for other issues.
    """

    path: str
    message: str
    suggestion: str | None = None
    code: str | None = None

    def describe(self) -> str:
        """The issue as one line, followed by the suggestion when there is one."""
        code = "" if self.code is None else f" [{self.code}]"
        suffix = "" if self.suggestion is None else f"\n  try: {self.suggestion}"
        return f"{self.path}: {self.message}{code}{suffix}"


class ValidationError(ValueError):
    """All problems found before opening an engine's data source.

    Raised by ``prepare`` and ``DataLoaderSpec.validate``; ``check`` returns
    the same issues without raising.

    :param issues: every issue found, each a `ValidationIssue`.
    """

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__(
            "invalid data-loader specification:\n"
            + "\n".join(f"- {issue.describe()}" for issue in self.issues)
        )


def _declares_time(value: Any) -> bool:
    if isinstance(value, (TimeWindow, ActiveInterval, EventTimeDelta)):
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            (
                item.name in {"time_column", "available_from"}
                and getattr(value, item.name) is not None
            )
            or _declares_time(getattr(value, item.name))
            for item in fields(value)
        )
    if isinstance(value, Mapping):
        return any(_declares_time(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_declares_time(item) for item in value)
    return False


def _immutable_declaration(value: Any, *, owned_mapping: bool = False) -> bool:
    """Only memoize declarations whose entire serialized tree is immutable.

    Some older constructors accept lists despite tuple annotations. Keep their
    fingerprints live rather than changing those constructors or caching stale IDs.
    """
    # Filter operands are intentionally opaque Any values. Their constructors
    # do not freeze nested state, so never cache contracts containing filters.
    if isinstance(value, RowFilter):
        return False
    if type(value) in (str, int, float, bool, type(None)) or isinstance(value, Enum):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(_immutable_declaration(item) for item in value)
    if isinstance(value, FrozenMapping):
        # Holds its own copy, so no outside owner can change it.
        return all(
            type(key) is str and _immutable_declaration(item, owned_mapping=True)
            for key, item in value.items()
        )
    if isinstance(value, MappingProxyType):
        # Only declaration Mapping fields copy their backing dictionaries.
        # A proxy passed through an opaque value (e.g. RowFilter.value) can
        # still be changed by its owner and must never make a cache eligible.
        return owned_mapping and all(
            type(key) is str and _immutable_declaration(item, owned_mapping=True)
            for key, item in value.items()
        )
    if is_dataclass(value) and not isinstance(value, type):
        if not type(value).__module__.startswith("kumo.pytorch."):
            return False
        return value.__dataclass_params__.frozen and all(
            _immutable_declaration(
                getattr(value, field.name),
                owned_mapping=str(field.type).startswith("Mapping["),
            )
            for field in fields(value)
        )
    return False


def sample_without_targets(sample: SampleSpec) -> SampleSpec:
    """The sample with every row-computed target removed, its own and its candidates'.

    Caller-supplied labels (`ExternalTarget`) read no rows, so their declarations stay.
    """

    def supplied(targets: Mapping[str, Any]) -> dict[str, Any]:
        return {name: value for name, value in targets.items() if isinstance(value, ExternalTarget)}

    return replace(
        sample,
        targets=supplied(sample.targets),
        candidate_sets={
            name: replace(candidates, targets=supplied(candidates.targets))
            for name, candidates in sample.candidate_sets.items()
        },
    )


@dataclass(frozen=True)
class DataLoaderSpec:
    """Schema, sampled values, and explicitly encoded model outputs.

    The complete, serializable contract. A model, optimizer, loss and training
    loop are not part of it.

    Validation happens in stages: constructors reject malformed declarations;
    ``resolve`` fills in types; ``validate`` checks dependencies, output
    compatibility and temporal rules without a dataset; ``check`` and
    ``prepare`` add the engine's own checks and source metadata;
    materialization checks actual values. Each refusal names the declaration
    and a next step.

    :param schema: stored tables, types, keys, relations and clocks.
    :param sample: what one example may read at its prediction time.
    :param batch: how selected values are encoded and arranged.
    :param version: contract format; unsupported formats are refused, not
        decoded heuristically.
    """

    schema: DataSchema
    sample: SampleSpec
    batch: BatchSpec
    version: int = 3

    def __post_init__(self) -> None:
        if self.version != 3:
            raise ValueError(f"unsupported data-loader specification version {self.version}")
        object.__setattr__(self, "_fingerprints_cacheable", _immutable_declaration(self))

    def validate(self) -> None:
        """Check the resolved contract without a dataset.

        :raises ValidationError: listing every issue found.
        """
        try:
            contract = self.resolve()
        except (ValueError, KeyError) as error:
            raise ValidationError(
                (
                    ValidationIssue(
                        "contract.types", str(error), "resolve column and calculation types first"
                    ),
                )
            ) from error
        issues = _Validator(contract).run()
        if issues:
            raise ValidationError(issues)

    def to_dict(self) -> dict[str, Any]:
        """The resolved contract as JSON-compatible data, with typed scalars made explicit."""
        return _encode(self.resolve())

    def resolve(self, *, dataset: Dataset | None = None) -> DataLoaderSpec:
        """Resolve authoring types before execution, serialization or identity.

        Parquet metadata is needed only for omitted stored types. The returned
        contract is self-contained; saved declarations are checked on rebinding.
        """
        unresolved = any(
            table.columns is None or any(dtype is None for dtype in table.columns.values())
            for table in self.schema.tables
        )
        if unresolved and dataset is None:
            raise ValueError(
                "contract has unresolved column types; call resolve(dataset=...) "
                "or save plan.contract after prepare()"
            )
        schema = dataset.resolve_schema(self.schema) if unresolved else self.schema
        sample = resolve_sample_types(schema, self.sample)
        return (
            self
            if schema is self.schema and sample is self.sample
            else replace(self, schema=schema, sample=sample)
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Sort named definitions while retaining explicit output/operand order."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataLoaderSpec:
        """Rebuild a contract from ``to_dict`` output; key placement is irrelevant."""
        decoded = _decode(dict(value))
        if not isinstance(decoded, cls):
            raise ValueError("serialized value is not a DataLoaderSpec")
        return decoded

    @classmethod
    def from_json(cls, value: str) -> DataLoaderSpec:
        """Rebuild a contract from ``to_json`` output."""
        return cls.from_dict(json.loads(value))

    @property
    def is_temporal(self) -> bool:
        """Whether any table has a clock or any sampled value depends on time.

        External labels alone do not make a contract temporal.
        """
        return bool(
            self.sample.events
            or self.sample.current_events
            or self.sample.dated_entity_values
            or any(not isinstance(value, ExternalTarget) for value in self.sample.targets.values())
            or _declares_time(self.sample)
            or any(table_clock(table) is not None for table in self.schema.tables)
        )

    @property
    def requires_anchor_time(self) -> bool:
        """Whether requests must carry ``as_of_time``; a non-temporal contract refuses one."""
        return self.is_temporal

    @property
    def input_fingerprint(self) -> str:
        """Identity of the inputs alone: the contract without targets or batching limits.

        Input-only fitted state (``FittedState.inputs``) is reused only under an
        equal input fingerprint.
        """
        cached = self.__dict__.get("_input_fingerprint")
        if cached is not None:
            return cached
        sample = replace(
            self.sample,
            targets={},
            candidate_sets={
                name: replace(value, targets={})
                for name, value in self.sample.candidate_sets.items()
            },
        )
        encodings, spaces, field_maps = set(), set(), set()
        for value in self.batch.inputs.values():
            groups = (
                value.representation.node_fields.values()
                if isinstance(value.representation, Graph)
                else (value.representation,)
            )
            for group in groups:
                if group.field_names is not None:
                    field_maps.add(group.field_names)
                spaces.add(group.field_space)
                spaces.add(getattr(group.row, "id_space", None))
                for use in group.fields:
                    spaces.add(use.id_space)
                    if isinstance(use.encoding, UseEncoding):
                        encodings.add(use.encoding.name)
        batch = replace(
            self.batch,
            targets={},
            batching=type(self.batch.batching)(),
            encodings={
                name: value for name, value in self.batch.encodings.items() if name in encodings
            },
            id_spaces={
                name: value for name, value in self.batch.id_spaces.items() if name in spaces
            },
            field_names={
                name: value for name, value in self.batch.field_names.items() if name in field_maps
            },
        )
        result = replace(self, sample=sample, batch=batch).fingerprint
        if self.__dict__["_fingerprints_cacheable"]:
            object.__setattr__(self, "_input_fingerprint", result)
        return result

    @property
    def fingerprint(self) -> str:
        """SHA-256 identity of the resolved contract.

        Placement of named definitions does not change it; explicit sequences
        (output fields, operands, composite components, markers) do.
        """
        cached = self.__dict__.get("_fingerprint")
        if cached is not None:
            return cached
        payload = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()
        result = "sha256:" + hashlib.sha256(payload).hexdigest()
        if self.__dict__["_fingerprints_cacheable"]:
            object.__setattr__(self, "_fingerprint", result)
        return result

    def explain(self) -> str:
        """A readable multi-line summary of the contract."""
        entity = self.sample.entity
        table = next((item for item in self.schema.tables if item.name == entity.table), None)
        key = table.primary_key if table else "<unknown table>"
        lines = [
            f"Data-loader specification v{self.version}",
            f"entity: {entity.table}.{key} ({entity.entity_type})",
        ]
        for name, value in self.sample.entity_values.items():
            if isinstance(value, Field):
                source = value.column
            elif isinstance(value, Compare):
                source = f"{value.left.operation} {value.operation} {value.right!r}"
            else:
                source = getattr(value, "operation", type(value).__name__)
            lines.append(f"entity value {name}: {source}")
        for name, event in self.sample.events.items():
            lines.append(f"history {name}: fields {', '.join(event.fields)}")
        for name, target in self.sample.targets.items():
            source = (
                f"{target.selection} future row"
                if isinstance(target, RowTarget)
                else f"{target.left.operation} {target.operation} {target.right!r}"
                if isinstance(target, Compare)
                else target.operation
                if isinstance(target, Aggregate)
                else f"external {target.dtype.name}"
            )
            lines.append(f"target {name}: {source}")
        for name, value in self.batch.inputs.items():
            representation = value.representation
            arrangement = (
                type(representation.row).__name__ if isinstance(representation, Fields) else "Graph"
            )
            lines.append(
                f"batch input {name}: {type(value.source).__name__} -> "
                f"{arrangement}, {type(value.layout).__name__}"
            )
        for name, value in self.batch.targets.items():
            lines.append(f"batch target {name}: {type(value).__name__}")
        return "\n".join(lines)


class _Validator:
    def __init__(self, contract: DataLoaderSpec):
        self.contract = contract
        self.schema, self.sample = contract.schema, contract.sample
        self.tables = {table.name: table for table in self.schema.tables}
        self.relations = {relation.name: relation for relation in self.schema.relations}
        self.issues: list[ValidationIssue] = []

    def add(
        self,
        path: str,
        message: str,
        suggestion: str = "correct the declaration",
    ) -> None:
        self.issues.append(ValidationIssue(path, message, suggestion))

    def run(self) -> tuple[ValidationIssue, ...]:
        self._sample(self.sample, "sample")
        if self.issues:
            return tuple(self.issues)
        from kumo.pytorch.contract.batch.validation import validate_batch

        self.issues.extend(validate_batch(self.contract))
        return tuple(self.issues)

    def _sample(self, sample: SampleSpec, path: str) -> None:
        entity = sample.entity
        if entity.table not in self.tables:
            self.add(f"{path}.entity.table", f"unknown table {entity.table!r}")
            return
        if entity.available_from is not None:
            self._time(entity.table, entity.available_from, f"{path}.entity.available_from")
        self._filters(entity.table, entity.filters, f"{path}.entity.filters")
        self._scope(sample.entity_values, entity.table, f"{path}.entity_values", event=False)
        for name, dated in sample.dated_entity_values.items():
            location = f"{path}.dated_entity_values.{name}"
            table = self._rows(dated.rows, entity.table, f"{location}.rows", "context")
            self._require_clock(dated.rows, table, f"{location}.rows")
            if isinstance(dated.field, Field):
                self._field(dated.field, table, f"{location}.field", True)
            if name in dated.fields:
                self.add(
                    f"{location}.fields", "an occurrence field duplicates its primary value name"
                )
            self._scope(
                {name: dated.field, **dated.fields}, table, f"{location}.fields", event=True
            )
        for collection, role in (("events", "context"), ("current_events", "current")):
            for name, event in getattr(sample, collection).items():
                location = f"{path}.{collection}.{name}"
                table = self._rows(event.rows, entity.table, f"{location}.rows", role)
                self._require_clock(event.rows, table, f"{location}.rows")
                self._scope(event.fields, table, f"{location}.fields", event=True)
                if role == "current" and event.limit != 1:
                    self.add(f"{location}.limit", "current events must explicitly set limit=1")
        for name, target in sample.targets.items():
            location = f"{path}.targets.{name}"
            if isinstance(target, ExternalTarget):
                continue
            if isinstance(target, RowTarget):
                table = self._rows(target.rows, entity.table, f"{location}.rows", "target")
                self._scope(target.fields, table, f"{location}.fields", event=True)
            else:
                self._aggregate(
                    target.left if isinstance(target, Compare) else target,
                    entity.table,
                    location,
                    "target",
                )
        for name, candidates in sample.candidate_sets.items():
            location = f"{path}.candidate_sets.{name}"
            self._sample(candidates.sample_spec(), location)
        self._graph(sample, path)

    def _scope(self, values: Mapping[str, Any], table: str, path: str, *, event: bool) -> None:
        try:
            _value_evaluation_order(values)
        except ValueError as error:
            self.add(path, str(error), "reference named values in the same scope without a cycle")
            return
        for name, value in values.items():
            location = f"{path}.{name}"
            reported = len(self.issues)
            if isinstance(value, Field):
                self._field(value, table, location, event)
            elif isinstance(value, (Aggregate, Compare)):
                self._aggregate(
                    value.left if isinstance(value, Compare) else value, table, location, "context"
                )
            elif isinstance(value, EventTimeDelta):
                if not event:
                    self.add(location, "event time is unavailable outside an event scope")
            elif not isinstance(value, (Constant, Derived)):
                self.add(location, f"unsupported sampled declaration {type(value).__name__}")
                continue
            if len(self.issues) > reported:
                continue
            try:
                value_dtype(self.schema, table, value, values)
            except (ValueError, KeyError, StopIteration) as error:
                self.add(f"{location}.dtype", str(error))

    def _column(self, table: str, column: str, path: str) -> DType | None:
        spec = self.tables.get(table)
        if spec is None or column not in spec.columns:
            self.add(path, f"unknown column {table}.{column}")
            return None
        return spec.columns[column]

    def _time(self, table: str, column: str, path: str) -> None:
        dtype = self._column(table, column, path)
        if dtype is not None and not dtype.temporal:
            self.add(
                path,
                f"time column {table}.{column} is {dtype.name}; "
                "comparing it with the request time requires timestamp or date32",
                f"name a timestamp or date32 column of {table!r}, or convert "
                f"{column!r} to one upstream",
            )

    def _filters(self, table: str, filters: Sequence[Any], path: str) -> None:
        for index, condition in enumerate(filters):
            location = f"{path}[{index}]"
            dtype = self._column(table, condition.column, f"{location}.column")
            if dtype is None or condition.operation in {"is_null", "not_null"}:
                continue
            literals = (
                condition.value if condition.operation in {"in", "not_in"} else (condition.value,)
            )
            try:
                for value in literals:
                    dtype.normalize(value, path=f"{location}.value")
                if condition.operation in {
                    "greater_than",
                    "greater_or_equal",
                    "less_than",
                    "less_or_equal",
                } and not (dtype.numeric or dtype.name in {"string", "date32", "timestamp"}):
                    raise ValueError(f"ordered comparison is unavailable for {dtype.name}")
            except (ValueError, TypeError) as error:
                self.add(f"{location}.value", str(error))

    def _destination(self, start: str, names: tuple[str, ...], path: str) -> str:
        table = start
        for index, name in enumerate(names):
            try:
                table = destination_table(self.schema, table, (name,))
            except ValueError as error:
                self.add(f"{path}[{index}]", str(error))
                return start
        return table

    def _rows(self, rows: Rows, start: str, path: str, role: str) -> str:
        table = self._destination(start, rows.path, f"{path}.path")
        self._filters(table, rows.filters, f"{path}.filters")
        clock = rows.time.column if rows.time else table_clock(self.tables[table])
        if clock is not None:
            self._time(table, clock, f"{path}.time.column")
        if rows.active is not None:
            self._time(table, rows.active.start_column, f"{path}.active.start_column")
            self._time(table, rows.active.end_column, f"{path}.active.end_column")
        if role == "context" and rows.time and (rows.time.stop is None or rows.time.stop > 0):
            self.add(f"{path}.time.stop", "context window must end at or before the sample time")
        elif role == "target" and (
            rows.time is None or rows.time.start is None or rows.time.start < 0
        ):
            self.add(
                f"{path}.time.start",
                "generated target window must start at or after the sample time",
            )
        elif role == "current" and (
            rows.time is None or rows.time.start != -1 or rows.time.stop != 0
        ):
            self.add(
                f"{path}.time",
                "current events require the exact window TimeWindow(..., start=-1, stop=0)",
            )
        return table

    def _require_clock(self, rows: Rows, table: str, path: str) -> None:
        if rows.time is None and table_clock(self.tables[table]) is None:
            self.add(path, "events require an explicit or destination-table clock")

    def _field(self, value: Field, start: str, path: str, event: bool) -> None:
        table = self._destination(start, value.path, f"{path}.path")
        self._column(table, value.column, f"{path}.column")
        clock = value.time_column or (table_clock(self.tables[table]) if value.path else None)
        if value.time_column is not None:
            self._time(table, value.time_column, f"{path}.time_column")
        if value.selection == "latest" and clock is None:
            self.add(
                f"{path}.selection",
                "latest field selection requires an explicit or destination-table clock",
            )
        if value.reference_time == "event" and not event:
            self.add(f"{path}.reference_time", "event time is unavailable for an entity field")

    def _aggregate(self, value: Aggregate, start: str, path: str, role: str) -> None:
        def source(source: Rows | GroupBy, location: str) -> None:
            if isinstance(source, Rows):
                self._rows(source, start, location, role)
                return
            source_fn(source.source, f"{location}.source")

        source_fn = source
        source(value.source, f"{path}.source")
        columns = self._source_columns(value.source, start, f"{path}.source")
        if columns is not None:
            if value.column is not None and value.column not in columns:
                self.add(f"{path}.column", f"unknown source column {value.column!r}")
            else:
                try:
                    reduction_dtype(value, columns)
                except ValueError as error:
                    self.add(f"{path}.dtype", str(error))
        if value.operation == "recency" and role == "target":
            self.add(
                f"{path}.operation",
                "'recency' measures elapsed time back from the request time and is historical-only",
            )

    def _source_columns(
        self, source: Rows | GroupBy, start: str, path: str
    ) -> Mapping[str, DType] | None:
        """The typed columns a source offers, or None once an unknown one is reported."""

        if isinstance(source, Rows):
            try:
                return source_column_types(self.schema, start, source)
            except ValueError:
                return None
        columns = self._source_columns(source.source, start, f"{path}.source")
        if columns is None:
            return None
        result: dict[str, DType] = {}
        for key in source.keys:
            if key not in columns:
                self.add(f"{path}.keys", f"unknown source column {key!r}")
                return None
            result[key] = columns[key]
        for name, reduction in source.values.items():
            location = f"{path}.values.{name}"
            if reduction.column is not None and reduction.column not in columns:
                self.add(f"{location}.column", f"unknown source column {reduction.column!r}")
                return None
            try:
                result[name] = reduction_dtype(reduction, columns)
            except ValueError as error:
                self.add(f"{location}.dtype", str(error))
                return None
        return result

    def _graph(self, sample: SampleSpec, path: str) -> None:
        for name, neighborhood in sample.neighborhoods.items():
            location = f"{path}.neighborhoods.{name}"
            frontier = {sample.entity.table}
            for layer_index, layer in enumerate(neighborhood.layers):
                destinations = set()
                for index, traversal in enumerate(layer.samples):
                    item = f"{location}.layers[{layer_index}].samples[{index}]"
                    relation = self.relations.get(traversal.relation)
                    if relation is None:
                        self.add(f"{item}.relation", f"unknown relation {traversal.relation!r}")
                        continue
                    start, end = (
                        (relation.left_table, relation.right_table)
                        if traversal.direction == "forward"
                        else (relation.right_table, relation.left_table)
                    )
                    if start not in frontier:
                        self.add(
                            f"{item}.direction",
                            f"traversal starts from {start!r}, "
                            f"not this frontier {sorted(frontier)}",
                        )
                    destinations.add(end)
                    clock = traversal.time_column or table_clock(self.tables[end])
                    if traversal.time_column:
                        self._time(end, traversal.time_column, f"{item}.time_column")
                    if traversal.temporal == "inherit":
                        if clock is None:
                            self.add(
                                f"{item}.temporal",
                                f"traversal into {end!r} has no clock to filter by, "
                                "and does not say it is intentionally non-temporal",
                                'declare TableSpec.time_column or set temporal="static" '
                                "for an intentionally non-temporal traversal",
                            )
                    if traversal.strategy == "recent" and clock is None:
                        self.add(
                            f"{item}.strategy",
                            f"'recent' sampling of {end!r} has no time column to order by",
                            "declare TableSpec.time_column or an explicit traversal time_column",
                        )
                frontier = destinations


def _known_types() -> dict[str, type[Any]]:
    from kumo.pytorch.contract import batch, dtypes, sample, schema

    result = {"DataLoaderSpec": DataLoaderSpec}
    for module in (batch, dtypes, sample, schema):
        for value in vars(module).values():
            if isinstance(value, type) and is_dataclass(value):
                result[value.__name__] = value
    return result


def _encode(value: Any) -> Any:
    from kumo.pytorch.encoding import json_value

    if is_dataclass(value) and not isinstance(value, type):
        arguments = {item.name: _encode(getattr(value, item.name)) for item in fields(value)}
        if type(value).__name__ == "Vocabulary" and value.values is not None:
            arguments["values"] = [
                _encode(item) for item in sorted(value.values, key=lambda item: item[1])
            ]
        return {"$type": type(value).__name__, **arguments}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _encode(item) for key, item in value.items()}
        entries = [(_encode(key), _encode(item)) for key, item in value.items()]
        entries.sort(key=lambda item: json.dumps(item[0], sort_keys=True))
        return {"$mapping": entries}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return json_value(value)
    if isinstance(value, (str, int, bool, float)) or value is None:
        return value
    encoded = json_value(value)
    if encoded is value:
        raise TypeError(f"cannot serialize {type(value).__name__}")
    return encoded


def _decode(value: Any) -> Any:
    from kumo.pytorch.encoding import unjson_value

    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    if not isinstance(value, dict):
        return value
    if "$type" in value:
        name = value["$type"]
        classes = _known_types()
        if name not in classes:
            raise ValueError(f"unknown serialized specification type {name!r}")
        return classes[name](
            **{key: _decode(item) for key, item in value.items() if key != "$type"}
        )
    if "$mapping" in value:
        return {_decode(key): _decode(item) for key, item in value["$mapping"]}
    if any(key in value for key in ("$bytes", "$decimal", "$date", "$timestamp", "$float")):
        return unjson_value(value)
    return {key: _decode(item) for key, item in value.items()}
