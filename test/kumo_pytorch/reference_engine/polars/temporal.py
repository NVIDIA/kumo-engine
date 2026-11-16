"""Batched relational selection and reductions executed by native Polars plans.

Only request metadata and root positions cross into Python. Joins, time filtering,
typed ordering, deduplication, grouping and reductions stay in Polars.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import numpy as np
import polars as pl
import pyarrow as pa
from kumo.pytorch.contract.sample import Aggregate, GroupBy, Rows, SampleRequest
from kumo.pytorch.contract.schema import table_clock
from kumo.pytorch.data.records import cutoff_to_epoch

_REDUCTIONS = frozenset(
    (
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
    )
)


def _epoch(column: str, dtype: pl.DataType) -> pl.Expr:
    value = pl.col(column)
    if isinstance(dtype, pl.Datetime):
        factor = {"ms": 1000, "us": 1_000_000, "ns": 1_000_000_000}[dtype.time_unit]
        ticks = value.cast(pl.Int64)
        # Stored instants ceil to seconds; Float64 would round ns boundaries.
        return ticks // factor + (ticks % factor != 0).cast(pl.Int64)
    if dtype == pl.Date:
        return value.cast(pl.Int64) * 86400
    return value.cast(pl.Int64, strict=True)


def _exact_sum_type(kind: pl.DataType) -> pl.DataType | None:
    # Arrow cannot export Polars Int128. Decimal128 carries wide exact integers
    # and preserves a decimal's scale; floating reductions remain Float64.
    if kind.is_integer() or kind == pl.Boolean:
        return pl.Decimal(38, 0)
    if isinstance(kind, pl.Decimal):
        return pl.Decimal(38, kind.scale)
    return None


class TemporalQueries:
    """Queries over one open PolarsStore; reset caches at each source build."""

    def __init__(self, store, schema):
        self.store = store
        self.tables = {table.name: table for table in schema.tables}
        self.relations = {relation.name: relation for relation in schema.relations}
        self._cache = {}
        self._bounds_cache = {}
        self._physical_names = {
            name for table in schema.tables for name in table.columns
        }
        self._namespace(self._physical_names)

    def _namespace(self, names):
        prefix = "__arena_"
        if any(name.startswith(prefix) for name in names):
            prefix += uuid4().hex + "_"
        for name in (
            "anchor",
            "time",
            "cutoff",
            "key",
            "tie",
            "result",
            "present",
            "mode_value",
            "frequency",
        ):
            setattr(self, "_" + name, prefix + name)
        self.anchor_column, self.time_column = self._anchor, self._time
        self._cache.clear()
        self._bounds_cache.clear()

    def reset_cache(self):
        self._cache.clear()
        self._bounds_cache.clear()

    def _open(self):
        if getattr(self.store, "_closed", False):
            raise RuntimeError("Polars temporal query store is closed")

    def _path(self, root_table, rows, directions):
        if root_table not in self.tables:
            raise ValueError(f"Unknown root table {root_table!r}")
        if not isinstance(rows, Rows):
            raise TypeError("select requires a Rows declaration")
        if rows.filters or rows.active is not None:
            raise ValueError(
                "row filters and active intervals are outside the selected DiskGraph scope"
            )
        if directions is not None and (
            len(directions) != len(rows.path)
            or any(direction not in {"forward", "reverse"} for direction in directions)
        ):
            raise ValueError(
                "directions must contain one forward/reverse entry per path hop"
            )
        current, hops = root_table, []
        for index, name in enumerate(rows.path):
            relation = self.relations.get(name)
            if relation is None:
                raise ValueError(f"Unknown relation {name!r}")
            direction = (
                directions[index]
                if directions is not None
                else ("forward" if current == relation.left_table else "reverse")
            )
            if direction == "forward":
                origin, source_key, destination, destination_key = (
                    relation.left_table,
                    relation.left_key,
                    relation.right_table,
                    relation.right_key,
                )
            else:
                origin, source_key, destination, destination_key = (
                    relation.right_table,
                    relation.right_key,
                    relation.left_table,
                    relation.left_key,
                )
            if current != origin:
                raise ValueError(
                    f"Relation {name!r} is disconnected from current table {current!r}"
                )
            if (
                index < len(rows.path) - 1
                and table_clock(self.tables[destination]) is not None
            ):
                raise ValueError(
                    f"Clocked intermediate table {destination!r} is outside the selected DiskGraph scope"
                )
            hops.append((source_key, destination, destination_key))
            current = destination
        return current, hops

    def _requests(self, request_key):
        return pl.DataFrame(
            {
                self._anchor: pl.Series(range(len(request_key)), dtype=pl.Int64),
                self._cutoff: pl.Series(
                    [cutoff for _, cutoff in request_key], dtype=pl.Int64
                ),
            }
        )

    def _columns(self, destination, columns):
        declared = self.tables[destination].columns
        if columns is None:
            return declared
        if isinstance(columns, (str, bytes)):
            raise TypeError(
                "columns must be a sequence of distinct declared column names"
            )
        columns = tuple(columns)
        if len(set(columns)) != len(columns) or set(columns) - set(declared):
            raise ValueError("columns must be distinct declared destination columns")
        return columns

    def _indexed_bounds(
        self, root_table, requests, rows, directions=None, *, cutoff_inclusive=True
    ):
        """Resolve single-hop time ranges without rescanning relation payloads."""
        self._open()
        destination, hops = self._path(root_table, rows, directions)
        if len(hops) != 1 or len(requests) > 65536:
            # Large fitting selections retain native bulk joins rather than
            # constructing per-request Python range state for the population.
            return None
        source_key, _, destination_key = hops[0]
        if (
            self.store._tables[root_table].schema[source_key]
            != self.store._tables[destination].schema[destination_key]
        ):
            return None
        definition = self.tables[destination]
        clock = rows.time.column if rows.time is not None else table_clock(definition)
        if clock is not None and clock not in definition.columns:
            raise ValueError(
                f"Time column {clock!r} is not declared by table {destination!r}"
            )
        request_key = tuple(
            (r.entity_id, cutoff_to_epoch(r.as_of_time)) for r in requests
        )
        cache_key = (
            root_table,
            request_key,
            repr(rows),
            None if directions is None else tuple(directions),
            cutoff_inclusive,
        )
        if cache_key in self._bounds_cache:
            return self._bounds_cache[cache_key]
        keys = self.store.fetch(
            root_table, [r.entity_id for r in requests], (source_key,)
        )[source_key].to_pylist()
        index = self.store.relation_index(destination, destination_key, clock)
        if index is None:
            return None
        lower = rows.time.start if rows.time is not None else None
        upper = rows.time.stop if rows.time is not None else 0

        def boundary(values, value, side="right"):
            # Convert only after bounding: NumPy may promote an out-of-range
            # Python integer to Float64 and lose adjacent int64 timestamps.
            if value < -(2**63):
                return 0
            if value > 2**63 - 1:
                return len(values)
            return int(np.searchsorted(values, np.int64(value), side=side))

        bounds = []
        for key, (_, cutoff) in zip(keys, request_key, strict=True):
            start, end = index["groups"].get(key, (0, 0))
            values = index["times"][start:end]
            left = (
                boundary(values, cutoff + lower)
                if clock is not None and lower is not None
                else 0
            )
            right = (
                boundary(
                    values, cutoff + upper, "right" if cutoff_inclusive else "left"
                )
                if clock is not None and upper is not None
                else len(values)
            )
            bounds.append((start + left, max(start + left, start + right)))
        result = (destination, index, bounds, request_key)
        self._bounds_cache[cache_key] = result
        return result

    def _indexed_frame(self, indexed, columns, *, order="ascending", limit=None):
        destination, index, bounds, request_key = indexed
        definition = self.tables[destination]
        columns = tuple(dict.fromkeys((*columns, definition.primary_key)))
        ranges = np.asarray(bounds, dtype=np.int64).reshape(-1, 2)
        starts, ends = ranges[:, 0], ranges[:, 1]
        counts = ends - starts
        if limit is not None:
            # A larger Python integer cannot trim any int64-sized range and
            # must not make NumPy promote its index arrays to object dtype.
            counts = np.minimum(counts, min(limit, np.iinfo(np.int64).max))
        offsets = np.cumsum(counts, dtype=np.int64) - counts
        owners = np.repeat(np.arange(len(bounds), dtype=np.int64), counts)
        within = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(
            offsets, counts
        )
        selected = (
            np.repeat(ends - 1, counts) - within
            if order == "descending"
            else np.repeat(starts, counts) + within
        )
        payload = self.store._tables[destination].select(columns)[
            index["positions"][selected]
        ]
        cutoffs = np.asarray([cutoff for _, cutoff in request_key], dtype=np.int64)
        metadata = pl.DataFrame(
            {
                self._anchor: owners,
                self._cutoff: cutoffs[owners],
                self._time: index["times"][selected],
            }
        )
        return metadata.hstack(payload)

    def _selected(
        self,
        root_table,
        requests,
        rows,
        directions=None,
        columns=None,
        *,
        cutoff_inclusive=True,
    ):
        self._open()
        destination, hops = self._path(root_table, rows, directions)
        # Cache semantics, never caller-container identity. Sources clear this
        # bounded-to-one-build cache; direct callers can reset_cache explicitly.
        request_key = tuple(
            (request.entity_id, cutoff_to_epoch(request.as_of_time))
            for request in requests
        )
        output_columns = self._columns(destination, columns)
        definition = self.tables[destination]
        clock = rows.time.column if rows.time is not None else table_clock(definition)
        if clock is not None and clock not in definition.columns:
            raise ValueError(
                f"Time column {clock!r} is not declared by table {destination!r}"
            )
        selected_columns = tuple(
            dict.fromkeys(
                (*output_columns, definition.primary_key, *((clock,) if clock else ()))
            )
        )
        key = (
            root_table,
            request_key,
            repr(rows),
            None if directions is None else tuple(directions),
            selected_columns,
            cutoff_inclusive,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return destination, cached
        indexed = self._indexed_bounds(
            root_table, requests, rows, directions, cutoff_inclusive=cutoff_inclusive
        )
        if indexed is not None:
            selected = self._indexed_frame(indexed, selected_columns)
            self._cache[key] = selected
            return destination, selected
        root = self.tables[root_table]
        lower = rows.time.start if rows.time is not None else None
        upper = rows.time.stop if rows.time is not None else 0
        # The store's PK-position lookup gathers selected roots instead of hashing
        # the whole root table for each request batch. Relations still use native joins.
        root_columns = (
            selected_columns
            if not hops
            else tuple(dict.fromkeys((root.primary_key, hops[0][0])))
        )
        roots = pl.from_arrow(
            self.store.fetch(root_table, [r.entity_id for r in requests], root_columns)
        )
        frame = self._requests(request_key).hstack(roots).lazy()
        for index, (source_key, table_name, destination_key) in enumerate(hops):
            final = index == len(hops) - 1
            columns = (
                tuple(dict.fromkeys((*selected_columns, destination_key)))
                if final
                else tuple(dict.fromkeys((destination_key, hops[index + 1][0])))
            )
            target = self.store._tables[table_name].lazy().select(columns)
            if final and clock is not None:
                target = target.with_columns(
                    _epoch(clock, self.store._tables[table_name].schema[clock]).alias(
                        self._time
                    )
                )
                # This coarse envelope cannot discard a row valid for any anchor.
                # Apply it before the join; retain exact per-anchor bounds below.
                keep = pl.col(self._time).is_not_null()
                if request_key:
                    if lower is not None:
                        keep &= pl.col(self._time).cast(pl.Int128) > pl.lit(
                            min(row[1] for row in request_key) + lower, dtype=pl.Int128
                        )
                    if upper is not None:
                        keep &= pl.col(self._time).cast(pl.Int128) <= pl.lit(
                            max(row[1] for row in request_key) + upper, dtype=pl.Int128
                        )
                target = target.filter(keep)
            frame = (
                frame.select(
                    self._anchor, self._cutoff, pl.col(source_key).alias(self._key)
                )
                .join(
                    target,
                    left_on=self._key,
                    right_on=destination_key,
                    how="inner",
                    coalesce=False,
                )
                .select(
                    self._anchor,
                    self._cutoff,
                    *columns,
                    *([self._time] if final and clock is not None else []),
                )
            )
        if clock is None:
            frame = frame.with_columns(pl.lit(0, dtype=pl.Int64).alias(self._time))
        else:
            if not hops:
                frame = frame.with_columns(
                    _epoch(clock, roots.schema[clock]).alias(self._time)
                )
            keep = pl.col(self._time).is_not_null()
            if lower is not None:
                keep &= pl.col(self._time).cast(pl.Int128) > pl.col(self._cutoff).cast(
                    pl.Int128
                ) + pl.lit(lower, dtype=pl.Int128)
            if upper is not None:
                time = pl.col(self._time).cast(pl.Int128)
                stop = pl.col(self._cutoff).cast(pl.Int128) + pl.lit(
                    upper, dtype=pl.Int128
                )
                keep &= (time <= stop) if cutoff_inclusive else (time < stop)
            frame = frame.filter(keep)
        if rows.distinct:
            frame = frame.unique(
                subset=[self._anchor, definition.primary_key], keep="any"
            )
        selected = frame.collect()
        self._cache[key] = selected
        return destination, selected

    def select(
        self,
        root_table: str,
        requests: Sequence[SampleRequest],
        rows: Rows,
        *,
        order: str = "ascending",
        limit: int | None = None,
        directions=None,
        columns=None,
        cutoff_inclusive: bool = True,
    ) -> pa.Table:
        """Select rows; graph cutoff policy applies before ordering and fanout."""
        if type(cutoff_inclusive) is not bool:
            raise TypeError("cutoff_inclusive must be a bool")
        if order not in {"ascending", "descending"}:
            raise ValueError("order must be ascending or descending")
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("limit must be a positive integer or None")
        indexed = self._indexed_bounds(
            root_table, requests, rows, directions, cutoff_inclusive=cutoff_inclusive
        )
        if indexed is not None:
            destination = indexed[0]
            columns = self._columns(destination, columns)
            frame = self._indexed_frame(indexed, columns, order=order, limit=limit)
        else:
            destination, frame = self._selected(
                root_table,
                requests,
                rows,
                directions,
                columns,
                cutoff_inclusive=cutoff_inclusive,
            )
            columns = self._columns(destination, columns)
            frame = frame.sort(
                [self._anchor, self._time, self.tables[destination].primary_key],
                descending=[False, order == "descending", order == "descending"],
                maintain_order=True,
            )
            if limit is not None:
                frame = frame.group_by(self._anchor, maintain_order=True).head(limit)
        output = frame.select(self._anchor, self._time, *columns).to_arrow()
        # Retain the declared Parquet field types instead of backend large-string variants.
        physical = self.store._schemas[destination]
        output_schema = pa.schema(
            [
                pa.field(self._anchor, pa.int64()),
                pa.field(self._time, pa.int64()),
                *(physical.field(name) for name in columns),
            ]
        )
        return output.cast(output_schema, safe=True)

    def lengths(self, root_table, requests, rows, *, limit=None) -> pa.Array:
        """Count selected rows natively without ordering or exporting row payloads."""
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("limit must be a positive integer or None")
        indexed = self._indexed_bounds(root_table, requests, rows)
        if indexed is not None:
            values = [
                end - start if limit is None else min(end - start, limit)
                for start, end in indexed[2]
            ]
            return pa.array(values, type=pa.int64())
        _, frame = self._selected(root_table, requests, rows, columns=())
        counts = frame.group_by(self._anchor).agg(
            pl.len().cast(pl.Int64).alias(self._result)
        )
        anchors = pl.DataFrame(
            {self._anchor: pl.Series(range(len(requests)), dtype=pl.Int64)}
        )
        result = anchors.join(
            counts, on=self._anchor, how="left", coalesce=True, maintain_order="left"
        )
        values = result.get_column(self._result).fill_null(0)
        if limit is not None:
            values = values.clip(upper_bound=limit)
        return values.to_arrow()

    def _reduction(self, frame, operation, column):
        if operation not in _REDUCTIONS:
            raise ValueError(f"Unsupported reduction {operation!r}")
        if operation == "count":
            return pl.len().cast(pl.Int64)
        if column not in frame.columns:
            raise ValueError(f"Reduction column {column!r} is not present")
        value = pl.col(column)
        if operation == "count_distinct":
            return value.drop_nulls().n_unique().cast(pl.Int64)
        if operation == "recency":
            return (
                pl.col(self._cutoff).cast(pl.Int128).first()
                - _epoch(column, frame.schema[column]).cast(pl.Int128).max()
            ).cast(pl.Float64)
        if operation in {"min", "max"}:
            return value.min() if operation == "min" else value.max()
        if operation == "mode":
            raise AssertionError("mode is evaluated by native grouped frequencies")
        if operation == "sum":
            kind = _exact_sum_type(frame.schema[column]) or pl.Float64
            numeric = (
                value.cast(pl.Int64) if frame.schema[column] == pl.Boolean else value
            )
            return (
                pl.when(value.count() > 0)
                .then(numeric.cast(kind, strict=True).sum())
                .otherwise(None)
            )
        numeric = value.cast(pl.Float64, strict=True)
        if operation == "mean":
            return numeric.mean()
        if operation.startswith("std_"):
            return numeric.std(ddof=0 if operation.endswith("population") else 1)
        return numeric.var(ddof=0 if operation.endswith("population") else 1)

    def _reduce_groups(self, frame, keys, reductions):
        expressions = [pl.col(self._cutoff).first().alias(self._cutoff)]
        modes, overflow_checks = [], []
        occupied = set(frame.columns) | set(keys) | {name for name, _, _ in reductions}
        for name, operation, column in reductions:
            if operation == "mode":
                if column not in frame.columns:
                    raise ValueError(f"Reduction column {column!r} is not present")
                modes.append((name, column))
            else:
                expressions.append(
                    self._reduction(frame, operation, column).alias(name)
                )
                kind = (
                    _exact_sum_type(frame.schema[column])
                    if operation == "sum"
                    else None
                )
                # Integer/bool source columns cannot overflow Decimal128 at any
                # feasible frame size; avoid an extra bound reduction in that path.
                physical = frame.schema[column] if kind is not None else None
                safely_bounded = (
                    physical == pl.Boolean
                    or physical is not None
                    and physical.is_integer()
                ) and frame.height <= (10**38 - 1) // (2**64)
                if kind is not None and not safely_bounded:
                    # Polars decimal/int128 SUM can wrap silently. Bound the
                    # worst intermediate sum using exact native integer arithmetic.
                    # Division avoids overflow in the bound itself. This deliberately
                    # refuses huge cancellation groups whose accumulator is unsafe.
                    check = self._result + "_overflow_" + str(len(overflow_checks))
                    while check in occupied:
                        check += "_"
                    occupied.add(check)
                    unscaled = pl.col(column).cast(kind).to_physical()
                    count = pl.col(column).count().cast(pl.Int128).clip(1, None)
                    expressions.append(
                        (
                            unscaled.abs().max()
                            > pl.lit(10**38 - 1, dtype=pl.Int128) // count
                        ).alias(check)
                    )
                    overflow_checks.append(check)
        result = frame.group_by(keys).agg(expressions)
        if overflow_checks:
            if result.select(
                pl.any_horizontal(
                    pl.col(name).fill_null(False) for name in overflow_checks
                ).any()
            ).item():
                raise OverflowError(
                    "Exact Polars sum cannot guarantee Decimal128(38, scale) bounds for this group"
                )
            result = result.drop(overflow_checks)
        for name, column in modes:
            value_name, frequency = self._mode_value, self._frequency
            value = pl.col(column)
            if frame.schema[column] in (pl.Float32, pl.Float64):
                # Native grouping equates signed zeros; choose the positive-zero
                # representative so later categorical text is row-order invariant.
                value = (
                    pl.when(value == 0)
                    .then(pl.lit(0, dtype=frame.schema[column]))
                    .otherwise(value)
                )
            counts = (
                frame.select(*keys, value.alias(value_name))
                .filter(pl.col(value_name).is_not_null())
                .group_by([*keys, value_name])
                .agg(pl.len().alias(frequency))
            )
            winners = (
                counts.sort(
                    [*keys, frequency, value_name],
                    descending=[*[False] * len(keys), True, False],
                    nulls_last=False,
                )
                .unique(subset=keys, keep="first")
                .select(*keys, pl.col(value_name).alias(name))
            )
            result = result.join(
                winners, on=keys, how="left", coalesce=True, nulls_equal=True
            )
        return result

    def _source(self, root_table, requests, source):
        if isinstance(source, Rows):
            return self._selected(root_table, requests, source)[1]
        if not isinstance(source, GroupBy):
            raise TypeError("Aggregate sources must be Rows or GroupBy")
        frame = self._source(root_table, requests, source.source)
        missing = set(source.keys) - set(frame.columns)
        if missing:
            raise ValueError(f"Group keys are not present: {sorted(missing)}")
        reductions = [
            (name, value.operation, value.column)
            for name, value in source.values.items()
        ]
        return self._reduce_groups(frame, [self._anchor, *source.keys], reductions)

    def aggregate_many(self, root_table, requests, aggregates):
        """Fuse reductions sharing a row selection; preserve opaque caller keys."""
        self._open()
        groups, output = {}, {}
        for name, aggregate in aggregates.items():
            if not isinstance(aggregate, Aggregate):
                raise TypeError("aggregate requires an Aggregate declaration")
            if aggregate.operation not in _REDUCTIONS:
                raise ValueError(f"Unsupported reduction {aggregate.operation!r}")
            if isinstance(aggregate.source, Rows):
                groups.setdefault(repr(aggregate.source), []).append((name, aggregate))
            else:
                output[name] = self.aggregate(root_table, requests, aggregate)
        for members in groups.values():
            rows = members[0][1].source
            if all(aggregate.operation == "count" for _, aggregate in members):
                counts = self.lengths(root_table, requests, rows)
                output.update((name, counts) for name, _ in members)
                continue
            columns = tuple(
                dict.fromkeys(
                    aggregate.column
                    for _, aggregate in members
                    if aggregate.operation != "count"
                )
            )
            _, frame = self._selected(root_table, requests, rows, columns=columns)
            reductions = [
                (self._result + "_" + str(i), aggregate.operation, aggregate.column)
                for i, (_, aggregate) in enumerate(members)
            ]
            grouped = self._reduce_groups(frame, [self._anchor], reductions)
            grouped = grouped.with_columns(pl.lit(True).alias(self._present)).select(
                self._anchor, self._present, *(alias for alias, _, _ in reductions)
            )
            anchors = pl.DataFrame(
                {self._anchor: pl.Series(range(len(requests)), dtype=pl.Int64)}
            )
            result = anchors.join(
                grouped,
                on=self._anchor,
                how="left",
                coalesce=True,
                maintain_order="left",
            )
            for (name, aggregate), (alias, _, _) in zip(
                members, reductions, strict=True
            ):
                values = result[alias]
                if aggregate.operation in {"count", "count_distinct", "sum"}:
                    values = result.select(
                        pl.when(pl.col(self._present).is_null())
                        .then(pl.lit(0).cast(values.dtype))
                        .otherwise(pl.col(alias))
                        .alias(alias)
                    )[alias]
                output[name] = values.to_arrow()
        return output

    def aggregate(
        self, root_table: str, requests: Sequence[SampleRequest], aggregate: Aggregate
    ) -> pa.Array:
        self._open()
        if not isinstance(aggregate, Aggregate):
            raise TypeError("aggregate requires an Aggregate declaration")
        if isinstance(aggregate.source, Rows):
            return self.aggregate_many(root_table, requests, {0: aggregate})[0]
        names = set(self._physical_names)
        source = aggregate.source
        while isinstance(source, GroupBy):
            names.update(source.keys)
            names.update(source.values)
            source = source.source
        if names.intersection(
            (
                self._anchor,
                self._time,
                self._cutoff,
                self._key,
                self._tie,
                self._result,
                self._present,
                self._mode_value,
                self._frequency,
            )
        ):
            self._namespace(names)
        frame = self._source(root_table, requests, aggregate.source)
        grouped = self._reduce_groups(
            frame,
            [self._anchor],
            [(self._result, aggregate.operation, aggregate.column)],
        )
        grouped = grouped.with_columns(pl.lit(True).alias(self._present)).select(
            self._anchor, self._result, self._present
        )
        anchors = pl.DataFrame(
            {
                self._anchor: pl.Series(
                    self._anchor, range(len(requests)), dtype=pl.Int64
                )
            }
        )
        result = anchors.join(grouped, on=self._anchor, how="left", coalesce=True).sort(
            self._anchor
        )
        if aggregate.operation in {"count", "count_distinct", "sum"}:
            result = result.with_columns(
                pl.when(pl.col(self._present).is_null())
                .then(pl.lit(0).cast(result.schema[self._result]))
                .otherwise(pl.col(self._result))
                .alias(self._result)
            )
        return result.get_column(self._result).to_arrow()
