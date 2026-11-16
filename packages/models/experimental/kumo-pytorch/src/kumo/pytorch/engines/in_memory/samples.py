# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build `BuiltSample`s from tables held as Python mappings.

Straightforward scans and joins, chosen so the file is easy to read. The
in-memory engine is a test tool and a toy for small data, so speed is not a goal.

This is not an `Engine` in the protocol sense — `InMemoryEngine` is. This is
what that engine ends up calling once `InMemorySource` has opened a dataset.
"""

from __future__ import annotations

import math
import struct
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from typing import Any

from kumo.pytorch.contract.sample import (
    ActiveInterval,
    Aggregate,
    BuiltSample,
    Compare,
    ComparisonOperation,
    Constant,
    DatedEntityValue,
    Derived,
    EntitySpec,
    EventSpec,
    EventTimeDelta,
    ExternalTarget,
    Field,
    Reduce,
    Rows,
    RowSource,
    RowTarget,
    SampleRequest,
    SampleSpec,
    TargetValue,
    _central_moment,
    _normalize_supplied_targets,
    _value_evaluation_order,
    dated_value_types,
    evaluate_derived,
    event_value_types,
    sample_value_types,
)
from kumo.pytorch.contract.schema import DataSchema, table_clock
from kumo.pytorch.data.records import (
    DatedValue,
    EntityHistory,
    EntityState,
    Event,
    FieldValue,
    Scalar,
    cutoff_to_epoch,
    timestamp_to_epoch,
)

__all__ = ["InMemorySampleBuilder"]


@dataclass(frozen=True)
class _Row:
    table: str
    values: Mapping[str, Any]
    ordinal: int


def _sort_value(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _reduction_key(value: Any) -> Any:
    # Raw float reductions use stored-bit identity, independently of vocabularies.
    return struct.pack("!d", value) if isinstance(value, float) else value


def _compare_values(left: Scalar, operation: ComparisonOperation, right: Scalar) -> bool:
    if operation == "equal":
        return left == right
    if operation == "not_equal":
        return left != right
    if operation == "greater_than":
        return left > right
    if operation == "greater_or_equal":
        return left >= right
    if operation == "less_than":
        return left < right
    if operation == "less_or_equal":
        return left <= right
    raise ValueError(f"unsupported comparison operation {operation!r}")


class InMemorySampleBuilder:
    """Execute sample recipes over small in-memory tables.

    The straightforward scans and joins are intentional: the in-memory engine
    favors plain code over speed.
    """

    def __init__(
        self,
        schema: DataSchema,
        tables: Mapping[str, Sequence[Mapping[str, Any]]],
    ):
        self.schema = schema
        self._table_specs = {table.name: table for table in schema.tables}
        self._relations = {relation.name: relation for relation in schema.relations}
        missing = set(self._table_specs) - set(tables)
        if missing:
            raise ValueError(f"missing configured tables: {sorted(missing)}")
        self._tables = {}
        for name, table in self._table_specs.items():
            rows = []
            for row_index, row in enumerate(tables[name]):
                missing_columns = set(table.columns) - row.keys()
                if missing_columns:
                    raise ValueError(
                        f"{name} row {row_index} is missing columns {sorted(missing_columns)}"
                    )
                rows.append(
                    _Row(
                        name,
                        {
                            column: dtype.normalize(
                                row[column], path=f"{name}.{column}[{row_index}]"
                            )
                            for column, dtype in table.columns.items()
                        },
                        row_index,
                    )
                )
            self._tables[name] = tuple(rows)
        self._indices: dict[tuple[str, str], dict[Any, tuple[_Row, ...]]] = {}
        self._validate_primary_keys()

    def build(
        self,
        recipe: SampleSpec,
        requests: Sequence[SampleRequest],
    ) -> list[BuiltSample]:
        """Build requested samples in request order."""

        sample_ids = [request.sample_id for request in requests]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample IDs must be unique")
        self._validate_recipe(recipe)
        return [self._build_one(recipe, request) for request in requests]

    def _validate_primary_keys(self) -> None:
        for table, spec in self._table_specs.items():
            if spec.primary_key is None:
                continue
            seen: set[Any] = set()
            for row in self._tables[table]:
                if spec.primary_key not in row.values:
                    raise ValueError(f"{table!r} row is missing primary key {spec.primary_key!r}")
                value = row.values[spec.primary_key]
                if value is None or value in seen:
                    raise ValueError(f"{table!r}.{spec.primary_key} must be unique and non-null")
                seen.add(value)

    def _validate_recipe(self, recipe: SampleSpec) -> None:
        _value_evaluation_order(recipe.entity_values)
        sample_value_types(self.schema, recipe)
        event_value_types(self.schema, recipe)
        event_value_types(self.schema, recipe, current=True)
        dated_value_types(self.schema, recipe)
        if recipe.entity.table not in self._table_specs:
            raise ValueError(f"unknown entity table {recipe.entity.table!r}")
        fields = [
            *(value for value in recipe.entity_values.values() if isinstance(value, Field)),
            *(
                value.field
                for value in recipe.dated_entity_values.values()
                if isinstance(value.field, Field)
            ),
            *(
                field
                for event in recipe.events.values()
                for field in event.fields.values()
                if isinstance(field, Field)
            ),
            *(
                field
                for event in recipe.current_events.values()
                for field in event.fields.values()
                if isinstance(field, Field)
            ),
            *(
                field
                for target in recipe.targets.values()
                if isinstance(target, RowTarget)
                for field in target.fields.values()
                if isinstance(field, Field)
            ),
        ]

        def row_leaves(source: RowSource) -> list[Rows]:
            return [source] if isinstance(source, Rows) else row_leaves(source.source)

        def validate_rows(rows: Rows, role: str) -> None:
            for relation in rows.path:
                if relation not in self._relations:
                    raise ValueError(f"unknown relation {relation!r}")
            if role == "context":
                if rows.time is not None and (rows.time.stop is None or rows.time.stop > 0):
                    raise ValueError("context rows may not extend after the sample time")
            elif role == "target":
                if rows.time is None or rows.time.start is None or rows.time.start < 0:
                    raise ValueError("generated target rows require a future time window")
            elif role == "current" and (
                rows.time is None or rows.time.start != -1 or rows.time.stop != 0
            ):
                raise ValueError("current event rows require the exact window (-1, 0]")

        for rows in [
            *(
                rows
                for value in recipe.entity_values.values()
                if isinstance(value, (Aggregate, Compare))
                for rows in row_leaves(
                    value.left.source if isinstance(value, Compare) else value.source
                )
            ),
            *(value.rows for value in recipe.dated_entity_values.values()),
            *(event.rows for event in recipe.events.values()),
        ]:
            validate_rows(rows, "context")
        for value in recipe.dated_entity_values.values():
            if self._rows_clock(recipe.entity.table, value.rows) is None:
                raise ValueError(
                    "dated entity values require an explicit or destination-table clock"
                )
        for event in recipe.events.values():
            if self._rows_clock(recipe.entity.table, event.rows) is None:
                raise ValueError("events require an explicit or destination-table clock")
        for event in recipe.current_events.values():
            validate_rows(event.rows, "current")
            if self._rows_clock(recipe.entity.table, event.rows) is None:
                raise ValueError("events require an explicit or destination-table clock")
        for neighborhood in recipe.neighborhoods.values():
            for layer in neighborhood.layers:
                for sample in layer.samples:
                    if sample.relation not in self._relations:
                        raise ValueError(f"unknown graph relation {sample.relation!r}")
        for target in recipe.targets.values():
            if isinstance(target, ExternalTarget):
                continue
            sources: list[Rows]
            if isinstance(target, (Aggregate, Compare)):
                aggregate = target.left if isinstance(target, Compare) else target
                sources = row_leaves(aggregate.source)
            else:
                sources = [target.rows]
            for rows in sources:
                validate_rows(rows, "target")
        for field in fields:
            for relation in field.path:
                if relation not in self._relations:
                    raise ValueError(f"unknown relation {relation!r}")
        for candidate in recipe.candidate_sets.values():
            self._validate_recipe(candidate.sample_spec())

    def _build_one(self, recipe: SampleSpec, request: SampleRequest) -> BuiltSample:
        targets = {
            target.name: target for target in _normalize_supplied_targets(recipe, request.targets)
        }
        cutoff = cutoff_to_epoch(request.as_of_time)
        entity_row = self._entity(recipe.entity, request.entity_id, cutoff)
        entity_values: list[FieldValue] = []
        for name in _value_evaluation_order(recipe.entity_values):
            expression = recipe.entity_values[name]
            if isinstance(expression, Field):
                entity_values.extend(
                    self._field_values(entity_row, name, expression, cutoff, event_time=None)
                )
                continue
            if isinstance(expression, Constant):
                value = FieldValue(name, expression.value, expression.dtype)
            elif isinstance(expression, Aggregate):
                value = self._aggregate(entity_row, name, expression, cutoff)
            elif isinstance(expression, Compare):
                value = self._compare_aggregate(entity_row, name, expression, cutoff)
            else:
                value = self._derived(entity_values, name, expression)
            if value is not None:
                entity_values.append(value)

        dated_values = self._dated_values(entity_row, recipe.dated_entity_values, cutoff)
        events = self._events(entity_row, recipe.events, cutoff)
        history = EntityHistory(
            entity_id=request.entity_id,
            evaluation_time=cutoff,
            entity=EntityState(tuple(entity_values), tuple(dated_values)),
            events=tuple(events),
            entity_type=recipe.entity.entity_type,
        )
        for name, target in recipe.targets.items():
            if isinstance(target, ExternalTarget):
                continue
            if isinstance(target, RowTarget):
                generated_row = self._row_target(entity_row, name, target, cutoff)
                if generated_row is not None:
                    targets[name] = TargetValue(name, generated_row)
            elif isinstance(target, Compare):
                generated = self._compare_aggregate(entity_row, name, target, cutoff)
                if generated is not None:
                    targets[name] = TargetValue(name, generated.value)
            else:
                generated = self._aggregate(entity_row, name, target, cutoff)
                if generated is not None:
                    targets[name] = TargetValue(name, generated.value)
        current_events = {
            name: tuple(
                self._events(
                    entity_row,
                    {name: event},
                    cutoff,
                    context=False,
                )
            )
            for name, event in recipe.current_events.items()
        }
        unknown_candidate_sets = set(request.candidates) - set(recipe.candidate_sets)
        if unknown_candidate_sets:
            raise ValueError(f"unknown candidate set(s): {sorted(unknown_candidate_sets)}")
        candidates = {}
        for group, candidate_spec in recipe.candidate_sets.items():
            requested = request.candidates.get(group, ())
            candidate_ids = [candidate.entity_id for candidate in requested]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"candidate IDs must be unique within set {group!r}")
            candidates[group] = tuple(
                self._build_one(
                    candidate_spec.sample_spec(),
                    SampleRequest(
                        sample_id=f"{request.sample_id}:{group}:{index}",
                        entity_id=candidate.entity_id,
                        as_of_time=cutoff,
                        targets=candidate.targets,
                    ),
                )
                for index, candidate in enumerate(requested)
            )
        metadata = {"split": request.split, **request.metadata}
        return BuiltSample(
            request.sample_id,
            history,
            tuple(targets.values()),
            current_events=current_events,
            metadata=metadata,
            candidates=candidates,
        )

    def _entity(self, spec: EntitySpec, entity_id: int, cutoff: int) -> _Row:
        key = self._table_specs[spec.table].primary_key
        entity_id = (
            self._table_specs[spec.table]
            .columns[key]
            .normalize(entity_id, path="requests.entity_id")
        )
        matches = self._index(spec.table, key).get(entity_id, ())
        if len(matches) != 1:
            raise ValueError(
                f"expected one {spec.table!r} row for {key}={entity_id!r}, found {len(matches)}"
            )
        entity = matches[0]
        if not all(
            condition.matches(
                entity.values, self._table_specs[entity.table].columns[condition.column]
            )
            for condition in spec.filters
        ):
            raise ValueError(f"entity {entity_id!r} does not satisfy its configured filters")
        if spec.available_from is not None:
            available = self._timestamp(entity, spec.available_from)
            if available >= cutoff:
                raise ValueError("entity did not exist before the requested sample time")
        return entity

    def _index(self, table: str, column: str) -> dict[Any, tuple[_Row, ...]]:
        key = (table, column)
        if key not in self._indices:
            grouped: dict[Any, list[_Row]] = defaultdict(list)
            for row in self._tables[table]:
                if column not in row.values:
                    raise ValueError(f"{table!r} row is missing relation column {column!r}")
                value = row.values[column]
                if value is not None:
                    grouped[value].append(row)
            self._indices[key] = {
                value: tuple(sorted(rows, key=self._row_sort_key))
                for value, rows in grouped.items()
            }
        return self._indices[key]

    def _row_sort_key(self, row: _Row) -> tuple[str, str | int]:
        if self._table_specs[row.table].primary_key is None:
            return "occurrence", row.ordinal
        return _sort_value(self._identity(row))

    def _identity(self, row: _Row) -> Any:
        primary_key = self._table_specs[row.table].primary_key
        return row.ordinal if primary_key is None else row.values[primary_key]

    def _follow(self, starting_rows: Sequence[_Row], path: tuple[str, ...]) -> list[_Row]:
        current = list(starting_rows)
        for relation_name in path:
            relation = self._relations[relation_name]
            followed: list[_Row] = []
            for row in current:
                if row.table == relation.left_table:
                    source_key = relation.left_key
                    target_table = relation.right_table
                    target_key = relation.right_key
                elif row.table == relation.right_table:
                    source_key = relation.right_key
                    target_table = relation.left_table
                    target_key = relation.left_key
                else:
                    raise ValueError(
                        f"relation {relation.name!r} cannot be followed from {row.table!r}"
                    )
                if source_key not in row.values:
                    raise ValueError(f"{row.table!r} row is missing relation column {source_key!r}")
                value = row.values[source_key]
                if value is not None:
                    followed.extend(self._index(target_table, target_key).get(value, ()))
            current = followed
        return current

    def _destination_table(self, start: str, path: tuple[str, ...]) -> str:
        table = start
        for relation_name in path:
            relation = self._relations[relation_name]
            if table == relation.left_table:
                table = relation.right_table
            elif table == relation.right_table:
                table = relation.left_table
            else:
                raise ValueError(f"relation {relation.name!r} cannot be followed from {table!r}")
        return table

    def _rows_clock(self, start: str, spec: Rows) -> str | None:
        """Resolve an explicit clock or inherit the final table's default."""

        if spec.time is not None:
            return spec.time.column
        destination = self._destination_table(start, spec.path)
        return table_clock(self._table_specs[destination])

    def _field_clock(self, start: str, spec: Field) -> str | None:
        if spec.time_column is not None or not spec.path:
            return spec.time_column
        destination = self._destination_table(start, spec.path)
        return table_clock(self._table_specs[destination])

    def _select_rows(self, entity: _Row, spec: Rows, cutoff: int) -> list[_Row]:
        selected = self._follow([entity], spec.path)
        if spec.distinct:
            unique: dict[tuple[str, Any], _Row] = {}
            for row in selected:
                identity = (row.table, self._identity(row))
                unique[identity] = row
            selected = list(unique.values())
        selected = [
            row
            for row in selected
            if all(
                condition.matches(
                    row.values, self._table_specs[row.table].columns[condition.column]
                )
                for condition in spec.filters
            )
        ]
        time_column = self._rows_clock(entity.table, spec)
        if time_column is not None:
            lower = (
                cutoff + spec.time.start
                if spec.time is not None and spec.time.start is not None
                else None
            )
            upper = (
                cutoff + spec.time.stop
                if spec.time is not None and spec.time.stop is not None
                else cutoff
                if spec.time is None
                else None
            )
            selected = [
                row
                for row in selected
                if (lower is None or self._timestamp(row, time_column) > lower)
                and (upper is None or self._timestamp(row, time_column) <= upper)
            ]
        if spec.active is not None:
            selected = [row for row in selected if self._active_at(row, spec.active, cutoff)]
        return sorted(selected, key=self._row_sort_key)

    def _active_at(self, row: _Row, interval: ActiveInterval, cutoff: int) -> bool:
        start = self._timestamp(row, interval.start_column)
        if interval.end_column not in row.values:
            raise ValueError(f"{row.table!r} row is missing {interval.end_column!r}")
        end = row.values[interval.end_column]
        return start < cutoff and (end is None or timestamp_to_epoch(end) > cutoff)

    def _timestamp(self, row: _Row, column: str) -> int:
        if column not in row.values or row.values[column] is None:
            raise ValueError(f"{row.table!r} row has no timestamp in {column!r}")
        return timestamp_to_epoch(row.values[column])

    def _field_values(
        self,
        base: _Row,
        name: str,
        spec: Field | Constant,
        sample_time: int,
        event_time: int | None,
    ) -> list[FieldValue]:
        if isinstance(spec, Constant):
            return [FieldValue(name, spec.value, spec.dtype)]
        candidates = self._follow([base], spec.path)
        time_column = self._field_clock(base.table, spec)
        if time_column is not None:
            reference = sample_time
            if spec.reference_time == "event":
                if event_time is None:
                    raise ValueError("event reference time is only valid for event fields")
                reference = event_time
            candidates = [
                row for row in candidates if self._timestamp(row, time_column) <= reference
            ]
        if spec.selection == "latest":
            if time_column is None:
                raise ValueError("latest field selection requires a destination table clock")
            candidates = sorted(
                candidates,
                key=lambda row: (
                    self._timestamp(row, time_column),
                    self._row_sort_key(row),
                ),
            )[-1:]
        elif spec.selection == "one" and len(candidates) > 1:
            raise ValueError(
                f"field {name!r} expected at most one related row, found {len(candidates)}"
            )
        else:
            candidates = sorted(candidates, key=self._row_sort_key)
        fields: list[FieldValue] = []
        for row in candidates:
            if spec.column not in row.values:
                raise ValueError(f"{row.table!r} row is missing field column {spec.column!r}")
            value = row.values[spec.column]
            if value is not None:
                fields.append(
                    FieldValue(name, value, self._table_specs[row.table].columns[spec.column])
                )
        return fields

    def _aggregate(
        self,
        entity: _Row,
        name: str,
        spec: Aggregate,
        cutoff: int,
    ) -> FieldValue | None:
        rows = self._evaluate_source(entity, spec.source, cutoff)
        result = self._reduce(rows, Reduce(spec.operation, spec.column, dtype=spec.dtype), cutoff)
        if result is None:
            return None
        return FieldValue(name, spec.dtype.normalize(result, path=f"aggregate.{name}"), spec.dtype)

    def _compare_aggregate(
        self,
        entity: _Row,
        name: str,
        spec: Compare,
        cutoff: int,
    ) -> FieldValue | None:
        value = self._aggregate(entity, name, spec.left, cutoff)
        if value is None:
            return None
        return FieldValue(
            name,
            _compare_values(value.value, spec.operation, spec.right),
            spec.dtype,
        )

    def _evaluate_source(
        self,
        entity: _Row,
        source: RowSource,
        cutoff: int,
    ) -> list[Mapping[str, Any]]:
        if isinstance(source, Rows):
            return [row.values for row in self._select_rows(entity, source, cutoff)]
        rows = self._evaluate_source(entity, source.source, cutoff)
        grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            try:
                key = tuple(row[column] for column in source.keys)
            except KeyError as error:
                raise ValueError(f"grouped row is missing key column {error.args[0]!r}") from error
            grouped[key].append(row)
        result: list[Mapping[str, Any]] = []
        for key in sorted(grouped, key=lambda value: tuple(_sort_value(item) for item in value)):
            output = dict(zip(source.keys, key, strict=True))
            for name, reduction in source.values.items():
                output[name] = reduction.dtype.normalize(
                    self._reduce(grouped[key], reduction, cutoff), path=f"group.{name}"
                )
            result.append(output)
        return result

    @staticmethod
    def _reduce(
        rows: Sequence[Mapping[str, Any]],
        spec: Reduce,
        cutoff: int,
    ) -> Any | None:
        if spec.operation == "count":
            return len(rows)
        values: list[Any] = []
        for row in rows:
            if spec.column not in row:
                raise ValueError(f"row is missing summary column {spec.column!r}")
            if row[spec.column] is not None:
                values.append(row[spec.column])
        if spec.operation == "count_distinct":
            return len({_reduction_key(value) for value in values})
        # An empty relation has the additive identity as its sum. This is
        # distinct from a non-empty relation whose selected values are all
        # null: there is then no observed value to sum, so the result is
        # missing like every other non-count reduction.
        if spec.operation == "sum" and not rows:
            zero = (
                Decimal(0)
                if spec.dtype.name.startswith("decimal")
                else 0.0
                if spec.dtype.name.startswith("float")
                else 0
            )
            return spec.dtype.normalize(zero, path="empty sum")
        if not values:
            return None
        if spec.operation == "recency":
            latest = max(timestamp_to_epoch(value) for value in values)
            return float(cutoff - latest)
        if spec.operation == "sum":
            if spec.dtype.name == "float64":
                total = 0.0
                for value in values:
                    total += float(value)
                return total
            if spec.dtype.integer:
                return sum(values)
            with localcontext() as context:
                context.prec = 160
                return sum(values)
        if spec.operation == "mean":
            return sum(float(value) for value in values) / len(values)
        if spec.operation == "min":
            return min(values)
        if spec.operation == "max":
            return max(values)
        if spec.operation == "mode":
            counts = Counter(_reduction_key(value) for value in values)
            highest = max(counts.values())
            return min(
                (value for value in values if counts[_reduction_key(value)] == highest),
                key=_sort_value,
            )
        if spec.operation == "var_population":
            return _central_moment([float(value) for value in values], ddof=0)
        if spec.operation == "var_sample":
            return _central_moment([float(value) for value in values], ddof=1)
        if spec.operation == "std_population":
            variance = _central_moment([float(value) for value in values], ddof=0)
            return None if variance is None else math.sqrt(variance)
        if spec.operation == "std_sample":
            variance = _central_moment([float(value) for value in values], ddof=1)
            return None if variance is None else math.sqrt(variance)
        raise ValueError(f"unsupported reduction operation {spec.operation!r}")

    @staticmethod
    def _derived(
        fields: Sequence[FieldValue],
        name: str,
        spec: Derived,
    ) -> FieldValue | None:
        available = {field.key: field.value for field in fields}
        if any(name not in available for name in spec.inputs):
            return None
        result = evaluate_derived(
            spec,
            available,
            path=f"derived.{name}",
            input_types={field.key: field.dtype for field in fields},
        )
        return None if result is None else FieldValue(name, result, spec.dtype)

    def _dated_values(
        self,
        entity: _Row,
        specs: Mapping[str, DatedEntityValue],
        cutoff: int,
    ) -> list[DatedValue]:
        values: list[DatedValue] = []
        for name, spec in specs.items():
            time_column = self._rows_clock(entity.table, spec.rows)
            if time_column is None:
                raise ValueError("dated entity values require a destination table clock")
            candidates: list[tuple[DatedValue, _Row]] = []
            for row in self._select_rows(entity, spec.rows, cutoff):
                timestamp = self._timestamp(row, time_column)
                for field in self._field_values(
                    row, name, spec.field, cutoff, event_time=timestamp
                ):
                    candidates.append(
                        (DatedValue(field.key, field.value, field.dtype, timestamp), row)
                    )
            if spec.first_per_value:
                first: dict[tuple[str, str], tuple[DatedValue, _Row]] = {}
                for value, row in sorted(candidates, key=lambda item: item[0].timestamp):
                    first.setdefault(_sort_value(value.value), (value, row))
                candidates = list(first.values())
            for value, row in candidates:
                fields = self._event_fields(
                    row,
                    spec.fields,
                    cutoff,
                    value.timestamp,
                    seed=(FieldValue(name, value.value, value.dtype),),
                )
                values.append(replace(value, fields=fields))
        return sorted(values, key=lambda item: (item.timestamp, item.key, _sort_value(item.value)))

    def _events(
        self,
        entity: _Row,
        specs: Mapping[str, EventSpec],
        cutoff: int,
        *,
        context: bool = True,
    ) -> list[Event]:
        events: list[tuple[tuple[Any, ...], Event]] = []
        for name, spec in specs.items():
            time_column = self._rows_clock(entity.table, spec.rows)
            if time_column is None:
                raise ValueError("events require a destination table clock")
            rows = sorted(
                self._select_rows(entity, spec.rows, cutoff),
                key=lambda row: (
                    self._timestamp(row, time_column),
                    self._row_sort_key(row),
                ),
                reverse=spec.order == "descending",
            )
            if spec.limit is not None:
                rows = rows[: spec.limit]
            for row in rows:
                timestamp = self._timestamp(row, time_column)
                fields = self._event_fields(row, spec.fields, cutoff, timestamp)
                events.append(
                    (
                        (timestamp, name, self._row_sort_key(row)),
                        Event(name, timestamp, tuple(fields)),
                    )
                )
        return [event for _key, event in sorted(events, key=lambda item: item[0])]

    def _row_target(
        self,
        entity: _Row,
        name: str,
        spec: RowTarget,
        cutoff: int,
    ) -> Event | None:
        time_column = self._rows_clock(entity.table, spec.rows)
        if time_column is None:
            raise ValueError("row targets require a destination table clock")
        rows = sorted(
            self._select_rows(entity, spec.rows, cutoff),
            key=lambda row: (
                self._timestamp(row, time_column),
                self._row_sort_key(row),
            ),
        )
        if not rows:
            return None
        row = rows[0] if spec.selection == "first" else rows[-1]
        timestamp = self._timestamp(row, time_column)
        return Event(name, timestamp, self._event_fields(row, spec.fields, cutoff, timestamp))

    def _event_fields(
        self,
        row: _Row,
        declarations: Mapping[str, Any],
        cutoff: int,
        timestamp: int,
        *,
        seed: tuple[FieldValue, ...] = (),
    ) -> tuple[FieldValue, ...]:
        values = list(seed)
        scope = {field.key: Constant(field.value, field.dtype) for field in seed}
        scope.update(declarations)
        for field_name in _value_evaluation_order(scope):
            if field_name not in declarations:
                continue
            field = declarations[field_name]
            if isinstance(field, Field):
                values.extend(
                    self._field_values(row, field_name, field, cutoff, event_time=timestamp)
                )
            elif isinstance(field, Constant):
                values.append(FieldValue(field_name, field.value, field.dtype))
            elif isinstance(field, EventTimeDelta):
                values.append(
                    FieldValue(
                        field_name, float(cutoff - timestamp) / field.seconds_per_unit, field.dtype
                    )
                )
            else:
                result = self._derived(values, field_name, field)
                if result is not None:
                    values.append(result)
        return tuple(values[len(seed) :])
