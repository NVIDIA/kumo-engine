# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native relational queries handed to Kumo as records or typed columnar batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from kumo.pytorch.contract.batch import EntitySource as EntityInputSource
from kumo.pytorch.contract.batch import EventSource, Input
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import (
    Aggregate,
    BuiltSample,
    Compare,
    Constant,
    Derived,
    EventTimeDelta,
    ExternalTarget,
    Field,
    SampleRequest,
    SampleSpec,
    TargetValue,
    _value_evaluation_order,
    destination_table,
    resolve_sample_types,
    value_dtype,
)
from kumo.pytorch.contract.schema import DataSchema
from kumo.pytorch.data.model import (
    ModelRowTarget,
    ModelSequence,
    ModelSourceBatch,
)
from kumo.pytorch.data.records import (
    EntityHistory,
    EntityState,
    Event,
    FieldValue,
    cutoff_to_epoch,
)
from kumo.pytorch.spec import ValidationError

from .availability import check_root_availability
from .encoding import VocabularyLookup, pack_token_stream
from .typed import derived_array, typed_array
from .validation import sample_issues


class EntityStore(Protocol):
    def fetch(
        self, table_name: str, entity_ids: Sequence[Any], columns: Sequence[str]
    ) -> pa.Table: ...
    def close(self) -> None: ...
    def describe(self) -> dict: ...


def _array(value):
    return value.combine_chunks() if isinstance(value, pa.ChunkedArray) else value


def _constant(field, size):
    return pa.repeat(pa.scalar(field.value, type=field.dtype.to_arrow()), size)


class EntitySource:
    vocabulary_lookup = VocabularyLookup
    pack_token_stream = staticmethod(pack_token_stream)

    def __init__(self, store: EntityStore, schema: DataSchema, queries):
        self._store: EntityStore | None = store
        schema = replace(
            schema,
            tables=tuple(
                replace(
                    t,
                    columns={
                        name: dtype
                        or DType.from_arrow(store._schemas[t.name].field(name).type)
                        for name, dtype in t.columns.items()
                    },
                )
                for t in schema.tables
            ),
        )
        self._tables = {table.name: table for table in schema.tables}
        self._schema = schema
        self._queries = queries
        self._validated_spec: SampleSpec | None = None
        from .graph import GraphExecutor

        self._graph = GraphExecutor(store, schema, queries)

    def _require_open(self) -> EntityStore:
        if self._store is None:
            raise RuntimeError("arena engine source is closed")
        return self._store

    def describe(self) -> dict:
        return {
            **self._require_open().describe(),
            "execution_semantics": {
                "event_ties": "native_typed_primary_key",
                "mode_ties": "smallest_native_value",
                "graph_edges": "unique_per_anchor",
                "graph_frontiers": "each_layer",
                "numerical_reductions": "native_with_documented_error_bounds",
            },
        }

    def _validate(self, spec: SampleSpec) -> None:
        if spec is not self._validated_spec:
            issues = sample_issues(spec, self._schema)
            if issues:
                raise ValidationError(issues)
            self._validated_spec = spec

    def _root_fields(self, spec, requests, cutoffs):
        store = self._require_open()
        fields = {
            name: field
            for name, field in spec.entity_values.items()
            if isinstance(field, (Field, Constant))
        }
        columns = tuple(
            dict.fromkeys(
                [f.column for f in fields.values() if isinstance(f, Field)]
                + ([spec.entity.available_from] if spec.entity.available_from else [])
            )
        )
        if not columns:
            columns = (self._tables[spec.entity.table].primary_key,)
        fetched = store.fetch(
            spec.entity.table, [r.entity_id for r in requests], columns
        )
        check_root_availability(spec, fetched, cutoffs)
        return {
            name: (
                typed_array(
                    fetched.column(f.column),
                    value_dtype(self._schema, spec.entity.table, f),
                )
                if isinstance(f, Field)
                else _constant(f, len(requests))
            )
            for name, f in fields.items()
        }

    def _offsets(self, table, size):
        owners = table.column(self._queries.anchor_column).to_numpy()
        counts = np.bincount(owners.astype(np.int64, copy=False), minlength=size)
        return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))

    @staticmethod
    def _reorder(table, offsets, reverse):
        if not reverse or not len(table):
            return table
        # Reverse within each contiguous anchor segment, including timestamp ties.
        counts = np.diff(offsets)
        indices = np.repeat(offsets[:-1] + offsets[1:] - 1, counts) - np.arange(
            len(table)
        )
        return table.take(pa.array(indices, type=pa.int64()))

    @staticmethod
    def _columns(fields):
        return tuple(
            dict.fromkeys(
                field.column
                for field in fields.values()
                if isinstance(field, Field) and field.column is not None
            )
        )

    def _fields(self, fields, table, cutoffs, destination):
        owners = _array(table.column(self._queries.anchor_column))
        timestamps = _array(table.column(self._queries.time_column))
        values = {}
        for name in _value_evaluation_order(fields):
            field = fields[name]
            if isinstance(field, EventTimeDelta):
                selected_cutoffs = pc.take(pa.array(cutoffs, type=pa.int64()), owners)
                try:
                    elapsed = pc.subtract_checked(selected_cutoffs, timestamps)
                except pa.ArrowInvalid:
                    # Preserve adjacent seconds at int64 extremes; converting
                    # operands to floating point before subtraction loses them.
                    wide = pa.decimal128(20, 0)
                    elapsed = pc.subtract(
                        pc.cast(selected_cutoffs, wide), pc.cast(timestamps, wide)
                    )
                values[name] = pc.divide(
                    pc.cast(elapsed, pa.float64(), safe=False),
                    float(field.seconds_per_unit),
                )
            elif isinstance(field, Field):
                values[name] = typed_array(
                    table.column(field.column),
                    value_dtype(self._schema, destination, field),
                )
            elif isinstance(field, Constant):
                values[name] = _constant(field, len(table))
            else:
                values[name] = derived_array(field, values, path=f"event.{name}")
        return values

    @staticmethod
    def _compare(spec, values):
        operation = {
            "greater_than": "greater",
            "less_than": "less",
            "greater_or_equal": "greater_equal",
            "less_or_equal": "less_equal",
        }.get(spec.operation, spec.operation)
        return getattr(pc, operation)(
            typed_array(values, spec.left.dtype),
            pa.scalar(spec.right, type=spec.left.dtype.to_arrow()),
        )

    def build_model(
        self, spec: SampleSpec, requests: Sequence[SampleRequest]
    ) -> ModelSourceBatch:
        return self.build_model_ordered(spec, requests)

    def build_model_ordered(self, spec, requests, *, event_orders=None):
        self._require_open()
        spec = resolve_sample_types(self._schema, spec)
        self._validate(spec)
        if any(r.candidates for r in requests):
            raise ValueError("candidate requests are outside the selected engine scope")
        if any(
            t.name in spec.targets
            and not isinstance(spec.targets[t.name], ExternalTarget)
            for r in requests
            for t in r.targets
        ):
            raise ValueError("a target was both supplied and generated")
        if event_orders is not None:
            unknown = set(event_orders) - set(spec.events)
            if unknown or any(
                not orders
                or any(order not in {"ascending", "descending"} for order in orders)
                for orders in event_orders.values()
            ):
                raise ValueError(
                    "event_orders requires known event names and ascending/descending orders"
                )
        cutoffs = tuple(cutoff_to_epoch(r.as_of_time) for r in requests)
        self._queries.reset_cache()
        try:
            values = self._root_fields(spec, requests, cutoffs)
            # One backend context can share relation selection across compatible
            # aggregates. Entity and target names occupy separate namespaces.
            aggregates = {
                ("entity", name): (
                    expression.left if isinstance(expression, Compare) else expression
                )
                for name, expression in spec.entity_values.items()
                if isinstance(expression, (Aggregate, Compare))
            }
            aggregates.update(
                {
                    ("target", name): (
                        expression.left
                        if isinstance(expression, Compare)
                        else expression
                    )
                    for name, expression in spec.targets.items()
                    if isinstance(expression, (Aggregate, Compare))
                }
            )
            aggregate_many = getattr(self._queries, "aggregate_many", None)
            reduced = (
                aggregate_many(spec.entity.table, requests, aggregates)
                if aggregate_many is not None
                else {
                    key: self._queries.aggregate(
                        spec.entity.table, requests, expression
                    )
                    for key, expression in aggregates.items()
                }
            )
            for name in _value_evaluation_order(spec.entity_values):
                expression = spec.entity_values[name]
                if isinstance(expression, Aggregate):
                    values[name] = typed_array(
                        reduced["entity", name], expression.dtype
                    )
                elif isinstance(expression, Compare):
                    values[name] = self._compare(expression, reduced["entity", name])
                elif isinstance(expression, Derived):
                    values[name] = derived_array(
                        expression, values, path=f"entity_values.{name}"
                    )
            events, ordered, row_targets = {}, {}, {}
            for name, event in spec.events.items():
                table = self._queries.select(
                    spec.entity.table,
                    requests,
                    event.rows,
                    order=event.order,
                    limit=event.limit,
                    columns=self._columns(event.fields),
                )
                offsets = self._offsets(table, len(requests))
                orders = (
                    ("ascending",)
                    if event_orders is None
                    else event_orders.get(name, ("ascending",))
                )
                for order in orders:
                    selected = self._reorder(table, offsets, event.order != order)
                    sequence = ModelSequence(
                        pa.array(offsets),
                        _array(selected.column(self._queries.time_column)),
                        self._fields(
                            event.fields,
                            selected,
                            cutoffs,
                            destination_table(
                                self._schema, spec.entity.table, event.rows.path
                            ),
                        ),
                        order,
                    )
                    events.setdefault(name, sequence)
                    ordered[name, order] = sequence
            targets = {}
            for i, request in enumerate(requests):
                for target in request.targets:
                    # setdefault evaluates its default even for existing keys;
                    # a population-sized default here makes fitting quadratic.
                    if target.name not in targets:
                        targets[target.name] = [None] * len(requests)
                    targets[target.name][i] = target.value
            for name, target in spec.targets.items():
                if isinstance(target, ExternalTarget):
                    values_for_target = targets.setdefault(name, [None] * len(requests))
                    targets[name] = pa.array(
                        values_for_target, type=target.dtype.to_arrow()
                    )
                elif isinstance(target, Aggregate):
                    targets[name] = typed_array(reduced["target", name], target.dtype)
                elif isinstance(target, Compare):
                    targets[name] = self._compare(target, reduced["target", name])
                else:
                    table = self._queries.select(
                        spec.entity.table,
                        requests,
                        target.rows,
                        order="ascending"
                        if target.selection == "first"
                        else "descending",
                        limit=1,
                        columns=self._columns(target.fields),
                    )
                    offsets = self._offsets(table, len(requests))
                    index = pa.array(
                        offsets[:-1], mask=np.diff(offsets) == 0, type=pa.int64()
                    )
                    fields = self._fields(
                        target.fields,
                        table,
                        cutoffs,
                        destination_table(
                            self._schema, spec.entity.table, target.rows.path
                        ),
                    )
                    row_targets[name] = ModelRowTarget(
                        pc.take(_array(table.column(self._queries.time_column)), index),
                        {n: pc.take(v, index) for n, v in fields.items()},
                    )
            return ModelSourceBatch(
                sample_ids=tuple(r.sample_id for r in requests),
                entity_ids=tuple(r.entity_id for r in requests),
                evaluation_times=cutoffs,
                entity_type=spec.entity.entity_type,
                entity_values={name: values[name] for name in spec.entity_values},
                events=events,
                ordered_events=ordered,
                targets=targets,
                row_targets=row_targets,
                splits=tuple(r.split for r in requests),
                request_metadata=tuple(
                    {"split": r.split, **r.metadata} for r in requests
                ),
            )
        finally:
            self._queries.reset_cache()

    def build(
        self, spec: SampleSpec, requests: Sequence[SampleRequest]
    ) -> list[BuiltSample]:
        spec = resolve_sample_types(self._schema, spec)
        batch = self.build_model_ordered(spec, requests)
        fields = {
            name: values.to_pylist() for name, values in batch.entity_values.items()
        }
        events = [[] for _ in requests]
        for name, sequence in batch.events.items():
            offsets = sequence.offsets.to_pylist()
            times = sequence.timestamps.to_pylist()
            columns = {n: v.to_pylist() for n, v in sequence.fields.items()}
            for owner in range(len(requests)):
                for row in range(offsets[owner], offsets[owner + 1]):
                    event = Event(
                        name,
                        times[row],
                        tuple(
                            FieldValue(
                                n,
                                v[row],
                                value_dtype(
                                    self._schema,
                                    destination_table(
                                        self._schema,
                                        spec.entity.table,
                                        spec.events[name].rows.path,
                                    ),
                                    spec.events[name].fields[n],
                                    spec.events[name].fields,
                                ),
                            )
                            for n, v in columns.items()
                            if v[row] is not None
                        ),
                    )
                    events[owner].append((times[row], name, row, event))
        targets = [dict((t.name, t) for t in r.targets) for r in requests]  # noqa: C402 -- preserve copied source
        for name, expression in spec.targets.items():
            if isinstance(expression, ExternalTarget):
                continue
            if isinstance(expression, (Aggregate, Compare)):
                for i, value in enumerate(batch.targets[name].to_pylist()):
                    if value is not None:
                        targets[i][name] = TargetValue(name, value)
            else:
                row = batch.row_targets[name]
                columns = {n: v.to_pylist() for n, v in row.fields.items()}
                for i, time in enumerate(row.timestamps.to_pylist()):
                    if time is not None:
                        event = Event(
                            name,
                            time,
                            tuple(
                                FieldValue(
                                    n,
                                    v[i],
                                    value_dtype(
                                        self._schema,
                                        destination_table(
                                            self._schema,
                                            spec.entity.table,
                                            expression.rows.path,
                                        ),
                                        expression.fields[n],
                                        expression.fields,
                                    ),
                                )
                                for n, v in columns.items()
                                if v[i] is not None
                            ),
                        )
                        targets[i][name] = TargetValue(name, event)
        return [
            BuiltSample(
                r.sample_id,
                EntityHistory(
                    r.entity_id,
                    batch.evaluation_times[i],
                    EntityState(
                        tuple(
                            FieldValue(
                                n,
                                v[i],
                                value_dtype(
                                    self._schema,
                                    spec.entity.table,
                                    spec.entity_values[n],
                                    spec.entity_values,
                                ),
                            )
                            for n, v in fields.items()
                            if v[i] is not None
                        ),
                        (),
                    ),
                    tuple(
                        item[-1]
                        for item in sorted(events[i], key=lambda item: item[:3])
                    ),
                    batch.entity_type,
                ),
                tuple(targets[i].values()),
                metadata=batch.request_metadata[i],
            )
            for i, r in enumerate(requests)
        ]

    def model_lengths(self, spec, input_spec, requests):
        self._require_open()
        self._validate(spec)
        if not isinstance(input_spec, Input):
            raise ValueError("typed length planning requires Input")  # noqa: TRY004 -- preserve reference exception
        entity_input = isinstance(input_spec.source, EntityInputSource)
        if entity_input or spec.entity.available_from:
            column = (
                spec.entity.available_from
                or self._tables[spec.entity.table].primary_key
            )
            fetched = self._require_open().fetch(
                spec.entity.table, [r.entity_id for r in requests], (column,)
            )
            check_root_availability(
                spec, fetched, tuple(cutoff_to_epoch(r.as_of_time) for r in requests)
            )
        if entity_input:
            return [len(input_spec.representation.fields)] * len(requests)
        if (
            not isinstance(input_spec.source, EventSource)
            or input_spec.source.collection != "history"
        ):
            raise ValueError(
                "typed length planning supports entity and historical event inputs"
            )
        counts = np.zeros(len(requests), dtype=np.int64)
        self._queries.reset_cache()
        try:
            for name in input_spec.source.events:
                event = spec.events[name]
                counts += self._queries.lengths(
                    spec.entity.table, requests, event.rows, limit=event.limit
                ).to_numpy(zero_copy_only=False)
        finally:
            self._queries.reset_cache()
        if input_spec.source.max_events is not None:
            counts = np.minimum(counts, input_spec.source.max_events)
        return counts.tolist()

    def fit_graph_fields(self, sample, spec, roots):
        self._require_open()
        self._validate(sample)
        return self._graph.fit_graph_fields(sample, spec, roots)

    def sample_graph(self, sample, spec, state, roots, seed, epoch):
        self._require_open()
        self._validate(sample)
        return self._graph.sample_graph(sample, spec, state, roots, seed, epoch)

    def close(self) -> None:
        store, self._store = self._store, None
        self._queries.reset_cache()
        if store is not None:
            store.close()
