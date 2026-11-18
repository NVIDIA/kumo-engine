# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reference row lookup that follows a `ServingRowsSpec` and nothing else.

It fetches, for one request, exactly what the spec's public data describes: each
route's join, time window and limit (every row tied at a limit's boundary kept),
hop by hop, then only the required tables and columns. Tests use it to check
that what the spec lists is enough to rebuild batch inference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from kumo.pytorch.api import DataSchema, DType, SampleRequest
from kumo.pytorch.rows import Clock
from kumo.pytorch.serving import RowLimit, RowsJoin, ServingRowsSpec, TableRoute

__all__ = ["Row", "Tables", "fetch"]

Row = Mapping[str, object]
Tables = Mapping[str, Sequence[Row]]


def fetch(
    spec: ServingRowsSpec, schema: DataSchema, tables: Tables, request: SampleRequest
) -> dict[str, list[dict[str, object]]]:
    """What a caller fetches for ``request`` by following ``spec`` over ``tables``."""

    keys = {table.name: table.primary_key for table in schema.tables}
    # Separate source occurrences can even be aliases of the same input mapping.
    sources = {
        name: [dict(row) for row in rows] if keys[name] is None else rows
        for name, rows in tables.items()
    }
    clock = Clock.of(schema)
    cutoff = clock.cutoff(request.as_of_time)
    fetched: dict[str, dict[object, Row]] = {}
    routes = sorted(
        ((table.name, route) for table in spec.tables.values() for route in table.routes),
        key=lambda item: item[1].hop,
    )
    for name, route in routes:
        rows = _joined(sources[name], route.join, request, fetched)
        ticks = _ticks(clock, next(table for table in schema.tables if table.name == name).columns)
        rows = _windowed(rows, route, cutoff, clock, ticks)
        if route.limit is not None:
            rows = _limited(rows, route.limit, route.join, ticks)
        bucket = fetched.setdefault(name, {})
        for row in rows:
            identity = row[keys[name]] if keys[name] is not None else id(row)
            bucket.setdefault(identity, row)
    return {
        name: [
            {column: row[column] for column, rows in table.columns.items() if rows.required}
            for row in fetched.get(name, {}).values()
        ]
        for name, table in spec.tables.items()
        if table.required
    }


def _joined(
    rows: Sequence[Row], join: RowsJoin, request: SampleRequest, fetched: dict[str, dict]
) -> list[Row]:
    if join.request_field == "entity_id":
        values = {request.entity_id}
    elif join.request_field is not None:
        set_name = join.request_field.removeprefix("candidates[").removesuffix("]").strip("'\"")
        values = {candidate.entity_id for candidate in request.candidates.get(set_name, ())}
    else:
        assert join.from_table is not None and join.from_column is not None
        values = {row[join.from_column] for row in fetched.get(join.from_table, {}).values()}
    values.discard(None)
    return [row for row in rows if row[join.column] in values]


def _ticks(clock: Clock, columns: Mapping[str, DType]) -> Callable[[Row, str], int]:
    def ticks(row: Row, column: str) -> int:
        if row[column] is None:
            raise ValueError("a time column the spec filters on has a NULL")
        return clock.stored(columns[column].normalize(row[column]))

    return ticks


def _windowed(
    rows: list[Row],
    route: TableRoute,
    cutoff: int,
    clock: Clock,
    ticks: Callable[[Row, str], int],
) -> list[Row]:
    if route.time_column is None:
        return rows
    start, stop = (None, 0) if route.window is None else (route.window.start, route.window.stop)
    return [
        row
        for row in rows
        if (start is None or ticks(row, route.time_column) > cutoff + clock.ticks(start))
        and (stop is None or ticks(row, route.time_column) <= cutoff + clock.ticks(stop))
    ]


def _limited(
    rows: list[Row], limit: RowLimit, join: RowsJoin, ticks: Callable[[Row, str], int]
) -> list[Row]:
    groups: dict[object, list[Row]] = {}
    for row in rows:
        groups.setdefault(None if limit.per == "request" else row[join.column], []).append(row)
    kept: list[Row] = []
    for members in groups.values():
        ordered = sorted(
            members,
            key=lambda row: ticks(row, limit.time_column),
            reverse=limit.order == "newest",
        )
        if len(ordered) > limit.count:
            boundary = ticks(ordered[limit.count - 1], limit.time_column)
            ordered = ordered[: limit.count] + [
                row for row in ordered[limit.count :] if ticks(row, limit.time_column) == boundary
            ]
        kept.extend(ordered)
    return kept
