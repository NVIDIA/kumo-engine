# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`Sampler`: entity values computed on the GPU for all of a call's anchors at once.

Sampled entity values, computed on the GPU for all of a call's anchors at once.
Every entity value kind upstream Kumo PyTorch defines for one scalar per anchor is built,
with Kumo's G-ROWS-001 and G-CALC-001 rules:

- Rows are selected by following the path from the entity row, relation by
  relation (forward from the relation's left table, backward otherwise); a NULL
  key matches nothing. Duplicates reached through several paths count once
  (``distinct``). Filters apply to the destination rows; the time window applies
  to the destination's clock: ``cutoff + start < time <= cutoff + stop``, every
  row at or before the cutoff without a window, and a selected row with no time
  is refused. ``active`` keeps rows with ``start < cutoff`` and an end after it
  (or none).
- ``GroupBy`` turns each anchor's rows into one row per distinct key
  combination (NULL is a key value), with named reductions; an ``Aggregate``
  reduces either kind of rows, and groupings nest.
- ``count`` and ``count_distinct`` (floats by their stored bits) are int64.
  ``sum`` of no rows is 0; of rows whose values are all NULL, NULL. Integer sums
  are exact (in two 32-bit halves), signed into int64 and unsigned into uint64,
  and refused when they do not fit. Float sums, ``mean``, variances and standard
  deviations are float64 (sample forms NULL under two values); ``min``/``max``
  keep the column's type; ``recency`` is the cutoff minus the latest time;
  ``mode`` breaks ties by the smallest (type name, printed value), as upstream.
  A float sum or mean of zeros is 0.0 (upstream adds from 0.0). Float
  ``min``/``max`` follow Python's over upstream's row order (printed primary
  key; groups by printed keys): NaN if the first value is, and of -0.0 and 0.0
  whichever comes first.
- A ``Field`` with a path (or its own clock) reads the related rows at or before
  the cutoff on the destination's clock: ``one`` refuses more than one,
  ``latest`` takes the newest, ties to the largest primary key by printed value
  (as upstream sorts rows). A NULL value, or no row, is NULL.
- ``Constant`` repeats its typed value. ``Derived`` is NULL when an operand is:
  integer arithmetic is exact in the declared type and refused when an operand
  or the result does not fit it; float arithmetic runs in float64 (float32
  results: operands rounded to float32 first, the result after, as upstream);
  ``divide`` by zero and ``log1p`` at or below -1 are NULL; ``abs`` of the
  smallest integer and ``sign`` of NaN are refused; calendar parts are UTC
  (Monday is weekday 0); ``MapValues`` matches members by value. A refusal's
  message is upstream's own, from ``evaluate_derived`` on the failing row.

Every anchor of a call is worked out at once: (anchor, row) pairs are expanded
hop by hop through key indexes built once per (table, column), and reductions
are segmented with libcudf's sort-based groupby, so the result does not depend
on row order or on the GPU's scheduling. Float sums add in the GPU's order, so
their last bits may differ from a row-by-row sum; ``log1p`` uses the GPU's own.
"""

from __future__ import annotations

from test.kumo_pytorch.reference_engine.gpu.data.text import (
    byte_order,
    is_text,
    text_column,
)
from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.spec import DataLoaderSpec

    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable

from test.kumo_pytorch.reference_engine.gpu.sampling.calculations import (
    _absolute,
    _arithmetic,
    _calendar,
    _divide,
    _log1p,
    _seconds,
    _sign,
)
from test.kumo_pytorch.reference_engine.gpu.sampling.filters import (
    _equals,
    _filter,
    _scalar,
    _text_codes,
)
from test.kumo_pytorch.reference_engine.gpu.sampling.host_values import (
    _column_python,
    _python,
    _python_at,
)
from test.kumo_pytorch.reference_engine.gpu.sampling.indexes import KeyIndex
from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
    _bits,
    _column_values,
    _count_distinct,
    _counts,
    _device_type,
    _exact_sum,
    _floats,
    _grouped,
    _most_frequent,
    _printed_ranks,
    _sum_refusal,
)
from test.kumo_pytorch.reference_engine.gpu.sampling.rows import (
    _Call,
    _Column,
    _GroupRows,
    _TableRows,
)

__all__ = ['Sampler']


class Sampler:
    """Computes a contract's entity values for anchors, on the GPU, against a plan's loaded tables."""

    def __init__(self, contract: DataLoaderSpec, tables: dict[str, GpuTable]) -> None:
        self.contract, self.tables = contract, tables
        self.specs = {table.name: table for table in contract.schema.tables}
        self.entity = contract.sample.entity.table
        self._indexes: dict[tuple[str, str], KeyIndex] = {}
        self._translations: dict[tuple[str, str, str, str], Any] = {}  # keys' identities on a hop's other side (`hop_keys`)
        self.graph_indexes: dict[tuple[str, str, str], Any] = {}  # test.kumo_pytorch.reference_engine.gpu.sampling.graphs.NeighborIndex by (table, key, clock)
        self._ranks: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
        self._dictionaries: dict[str, Any] = {}

    def values(self, names: list[str], entity_rows: Any, cutoffs: Any,
               flags: list[tuple[Any, Any]]) -> dict[str, Values]:
        """Each named entity value (and what it is calculated from) for the anchors whose entity rows and
        cutoffs are given. A check's message may be a callable, called only when the check fails."""

        call = _Call(self.contract.sample.entity_values, self.entity, entity_rows, cutoffs, flags)
        return {name: self._named(name, call) for name in names}

    def scope_values(self, names: list[str], scope: Any, table: str, rows: Any, cutoffs: Any, times: Any,
                     flags: list[tuple[Any, Any]]) -> dict[str, Values]:
        """Named values of one event component's `scope` (its fields), for events at `rows` of `table`, with
        each event's anchor cutoff and time."""

        call = _Call(scope, table, rows, cutoffs, flags, times)
        return {name: self._named(name, call) for name in names}

    # -- one value ------------------------------------------------------------------------------

    def _named(self, name: str, call: _Call) -> Values:
        if name not in call.computed:
            call.computed[name] = self._value(name, call.scope[name], call)
        return call.computed[name]

    def _value(self, name: str, spec: Any, call: _Call) -> Values:
        import cupy

        from kumo.pytorch.contract.sample import (
            Aggregate,
            Compare,
            Constant,
            Derived,
            EventTimeDelta,
            Field,
        )

        if isinstance(spec, Field):
            return self._field(name, spec, call)
        if isinstance(spec, Constant):
            return self._constant(name, spec, call.rows.size)
        if isinstance(spec, Aggregate):
            return self._aggregate(name, spec, call)
        if isinstance(spec, Compare):
            return self._compare(spec, self._aggregate(name, spec.left, call))
        if isinstance(spec, Derived):
            return self._derived(name, spec, call)
        if isinstance(spec, EventTimeDelta):  # upstream: float(cutoff - time) / seconds per unit
            return Values(spec.dtype, (call.cutoffs - call.times).astype(cupy.float64) / spec.seconds_per_unit)
        raise ValueError(f"test.kumo_pytorch.reference_engine.gpu does not build {type(spec).__name__} values yet")

    def _field(self, name: str, spec: Any, call: _Call) -> Values:
        import cupy

        from kumo.pytorch.contract.schema import table_clock

        if not spec.path and spec.time_column is None:
            return _column_values(self.tables[call.table], spec.column, call.rows)
        owner, rows, table_name = self._follow(spec.path, call.table, call.rows)
        table = self.tables[table_name]
        clock = spec.time_column or (table_clock(self.specs[table_name]) if spec.path else None)
        times = None
        if clock is not None:
            times = self._times(table, clock, rows, call.flags)
            reference = call.times if spec.reference_time == "event" else call.cutoffs  # an event's own time
            keep = times <= reference[owner]
            owner, rows, times = owner[keep], rows[keep], times[keep]
        count = call.rows.size
        if spec.selection == "one":
            found = _counts(owner, count)
            many = found > 1
            call.flags.append((lambda: f"field {name!r} expected at most one related row, found "
                               f"{int(found[cupy.flatnonzero(many)[0]])}", many.any()))
            picked = cupy.full(count, -1, dtype=cupy.int64)
            picked[owner] = rows
        else:
            picked = self._latest(table, owner, rows, times, count)
        return _column_values(table, spec.column, picked, missing=True)

    def _latest(self, table: GpuTable, owner: Any, rows: Any, times: Any, count: int) -> Any:
        """Each anchor's newest row; ties to the largest primary key by printed value, as upstream sorts rows."""

        from test.kumo_pytorch.reference_engine.gpu.data.printed import (
            key_columns,
            on_gpu,
        )
        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
            _column,
            _view,
        )

        import cupy
        import pylibcudf as plc

        picked = cupy.full(count, -1, dtype=cupy.int64)
        if rows.size == 0:
            return picked
        key = self.specs[table.name].primary_key
        columns = [owner, cupy.ascontiguousarray(times)]
        if on_gpu(table.dtypes[key]):
            columns += key_columns(table, key, rows)
        order = _view(plc.sorting.stable_sorted_order(plc.Table([_column(column) for column in columns]),
                                                      [plc.types.Order.ASCENDING] * len(columns),
                                                      [plc.types.NullOrder.AFTER] * len(columns)))
        owner, rows, times = owner[order], rows[order], times[order]
        last = cupy.concatenate([owner[1:] != owner[:-1], cupy.ones(1, dtype=bool)])
        picked[owner[last]] = rows[last]
        if not on_gpu(table.dtypes[key]):  # ties at the newest time: the largest printed key, worked out for them
            newest = cupy.full(count, cupy.iinfo(cupy.int64).min, dtype=cupy.int64)
            newest[owner[last]] = times[last]
            tied = times == newest[owner]
            tied_counts = cupy.zeros(count, dtype=cupy.int64)
            cupy.add.at(tied_counts, owner[tied], 1)
            at = cupy.flatnonzero(tied & (tied_counts[owner] > 1))
            if at.size:
                rank = key_columns(table, key, rows[at])[0]
                best = cupy.full(count, -1, dtype=cupy.int64)
                cupy.maximum.at(best, owner[at], rank)
                chosen = at[rank == best[owner[at]]]
                picked[owner[chosen]] = rows[chosen]  # among equal printed keys, the last in order, as upstream
        return picked

    def _constant(self, name: str, spec: Any, count: int) -> Values:
        import cupy

        dtype, value = spec.dtype, spec.value
        present = cupy.full(count, value is not None, dtype=bool)
        if dtype.name.startswith("decimal"):
            from test.kumo_pytorch.reference_engine.gpu.data.decimals import (
                constant_words,
            )

            import pyarrow as pa

            number = 0.0 if value is None else float(value)  # float(Decimal): upstream's rounding
            return Values(dtype, None, present, host=pa.array([value] * count, dtype.to_arrow()),
                          numbers=cupy.full(count, number, dtype=cupy.float64),
                          words=constant_words(value, dtype.scale, count))
        if is_text(dtype):
            return Values(dtype, cupy.zeros(count, dtype=cupy.int32), present, self._texts(repr(spec), [value]))
        stored = 0 if value is None else _scalar(dtype, value)
        return Values(dtype, cupy.full(count, stored, dtype=_device_type(dtype)), present)

    def _aggregate(self, name: str, spec: Any, call: _Call) -> Values:
        rows = self._source(spec.source, call)
        column = None if spec.operation == "count" else rows.column(spec.column)
        return self._reduce(f"aggregate.{name}", spec.operation, column, spec.dtype, rows.owner, call.rows.size,
                            call.cutoffs, call.flags)

    def _compare(self, spec: Any, left: Values) -> Values:
        import cupy

        from kumo.pytorch.contract.dtypes import DType

        right = left.dtype.normalize(spec.right, path="compare.right")
        values, right_value = left.values, right
        if left.texts is not None:  # text codes: compare by code, against the code a text would take
            values, right_value = _text_codes(left.texts, values, right)
        elif left.dtype.name.startswith("decimal"):  # exactly: each value's order against the right one's
            from test.kumo_pytorch.reference_engine.gpu.data.decimals import (
                compare,
                constant_words,
                words,
            )

            values, right_value = compare(words(left), constant_words(right, left.dtype.scale)), 0
        else:
            right_value = _scalar(left.dtype, right)
        operations = {"equal": cupy.equal, "not_equal": cupy.not_equal, "greater_than": cupy.greater,
                      "greater_or_equal": cupy.greater_equal, "less_than": cupy.less,
                      "less_or_equal": cupy.less_equal}
        result = operations[spec.operation](values, right_value)
        return Values(spec.dtype or DType("bool"), result, left.present)

    # -- calculations ---------------------------------------------------------------------------

    def _derived(self, name: str, spec: Any, call: _Call) -> Values:
        from test.kumo_pytorch.reference_engine.gpu.data.decimals import (
            at_most,
            calculate,
            decimal_values,
            is_decimal,
            words,
        )

        from kumo.pytorch.contract.sample import CalendarPart, MapValues

        where = f"derived.{name}"
        operation, dtype = spec.operation, spec.dtype
        given = [self._named(input_name, call) for input_name in spec.inputs]
        # decimals: exact into a decimal result (and mapped), else their float64 numbers, as upstream's float(value)
        exact = is_decimal(dtype) or isinstance(operation, MapValues)
        operands = [_floats(operand) if is_decimal(operand.dtype) and not exact else operand for operand in given]
        for operand in operands:
            if operand.values is None and not is_decimal(operand.dtype):
                raise ValueError(f"{where}: test.kumo_pytorch.reference_engine.gpu does not calculate with {operand.dtype.name} values yet")
        valid = operands[0].valid()
        for operand in operands[1:]:
            valid = valid & operand.valid()
        if isinstance(operation, CalendarPart):
            return Values(dtype, _calendar(operation.part, operands[0]), valid)
        if isinstance(operation, MapValues):
            return self._map(name, spec, operands[0], valid, call)
        bad = None
        if is_decimal(dtype):  # add, subtract, multiply, abs: exact, then checked against the result's type
            result, bad = calculate(operation, operands, dtype)
        elif operation in ("add", "subtract", "multiply"):
            result, bad = _arithmetic(operation, operands, dtype)
        elif operation in ("divide", "log1p"):
            result, null = _divide(operands) if operation == "divide" else _log1p(operands[0])
            if operation == "log1p" and is_decimal(given[0].dtype):
                # NULL at or below -1 exactly; above it by less than half a float step, upstream's
                # math.log1p(float(x)) is math.log1p(-1.0), which raises
                exactly = at_most(words(given[0]), given[0].dtype.scale, -1)
                bad, null = null & ~exactly, exactly
            valid = valid & ~null
        elif operation == "abs":
            result, bad = _absolute(operands[0], dtype)
        else:
            result, bad = _sign(operands[0])
        if bad is not None:
            bad = bad & valid
            call.flags.append((lambda: self._refusal(spec, where, given, bad, call), bad.any()))
        if is_decimal(dtype):
            return decimal_values(dtype, result, valid)
        return Values(dtype, result, valid)

    def _map(self, name: str, spec: Any, operand: Values, valid: Any, call: _Call) -> Values:
        """`MapValues`: each operand to its group's output; unknown values by the declared policy."""

        from test.kumo_pytorch.reference_engine.gpu.data.decimals import (
            constant_words,
            decimal_values,
            is_decimal,
        )

        import cupy

        from kumo.pytorch.contract.sample import _typed_map_groups

        operation, dtype, where = spec.operation, spec.dtype, f"derived.{name}"
        groups = _typed_map_groups(operation, operand.dtype)
        outputs = [dtype.normalize(output, path="map result") for output, _ in groups]
        fallback = (dtype.normalize(operation.fallback, path="map fallback") if operation.unknown == "fallback"
                    else None)
        chosen = cupy.full(operand.size, len(groups), dtype=cupy.int64)  # the last slot: not in a group
        for index, (_, members) in enumerate(groups):
            hit = cupy.zeros(operand.size, dtype=bool)
            for member in members:
                hit |= _equals(operand, member)
            chosen = cupy.where(hit, index, chosen)
        if operation.unknown == "error":
            bad = valid & (chosen == len(groups))
            call.flags.append((lambda: self._refusal(spec, where, [operand], bad, call), bad.any()))
        slots = [*outputs, fallback]
        present = cupy.asarray([value is not None for value in slots], dtype=bool)
        if is_decimal(dtype):  # each slot's words, picked
            table = cupy.concatenate([constant_words(value, dtype.scale) for value in slots])
            return decimal_values(dtype, table[chosen], valid & present[chosen])
        texts = None
        if is_text(dtype):
            texts = self._texts(repr(spec), slots)
            ordered = sorted({text for text in slots if text is not None}, key=byte_order)
            stored = [0 if text is None else ordered.index(text) for text in slots]
        else:
            stored = [0 if value is None else _scalar(dtype, value) for value in slots]
        table = cupy.asarray(stored, dtype=_device_type(dtype))
        return Values(dtype, table[chosen], valid & present[chosen], texts)

    def _refusal(self, spec: Any, where: str, operands: list[Values], bad: Any, call: _Call) -> str:
        """Upstream's own message for the first refused row: `evaluate_derived` on its Python values (a constant's
        as declared, as upstream holds it)."""

        import cupy

        from kumo.pytorch.contract.sample import Constant, evaluate_derived

        row = int(cupy.flatnonzero(bad)[0])
        values = {name: call.scope[name].value if isinstance(call.scope.get(name), Constant) else
                  _python_at(operand, row) for name, operand in zip(spec.inputs, operands, strict=True)}
        types = {name: operand.dtype for name, operand in zip(spec.inputs, operands, strict=True)}
        try:
            evaluate_derived(spec, values, path=where, input_types=types)
        except ValueError as error:
            return str(error)
        return f"{where}: test.kumo_pytorch.reference_engine.gpu refused {values}, which upstream Kumo PyTorch accepts (a bug in test.kumo_pytorch.reference_engine.gpu)"

    def _texts(self, key: str, texts: list[Any]) -> Any:
        """The sorted distinct texts a calculated text value can take, built once per declaration (`key`: its
        repr; names repeat across scopes)."""

        if key not in self._dictionaries:
            self._dictionaries[key] = text_column(sorted({text for text in texts if text is not None},
                                                         key=byte_order))
        return self._dictionaries[key]

    # -- selecting rows -------------------------------------------------------------------------

    def _index(self, table: str, column: str) -> KeyIndex:
        key = (table, column)
        if key not in self._indexes:
            self._indexes[key] = KeyIndex(self.tables[table], column)
        return self._indexes[key]

    def _follow(self, path: tuple[str, ...], table: str, base_rows: Any) -> tuple[Any, Any, str]:
        """(base, row) pairs reached from each base row of `table` along `path`, grouped by base, with repeats.

        Keys match as upstream's Python values compare (`test.kumo_pytorch.reference_engine.gpu.data.keys`, `hop_keys`).
        """

        from test.kumo_pytorch.reference_engine.gpu.sampling.paths import (
            _Routes,
        )

        import cupy

        owner = cupy.arange(base_rows.size, dtype=cupy.int64)
        rows = base_rows.astype(cupy.int64)
        routes, route = _Routes(self.contract.schema), ()
        for name in path:
            route = routes.hop(table, route, name)
            hop = route[-1]
            present = self.tables[hop.from_table].present(hop.from_column)
            if present is not None:
                keep = present[rows]
                owner, rows = owner[keep], rows[keep]
            found = self.hop_keys(hop.from_table, hop.from_column, rows, hop.table, hop.column)
            table = hop.table
            if found is None:
                owner, rows = cupy.zeros(0, dtype=cupy.int64), cupy.zeros(0, dtype=cupy.int64)
                continue
            keys, valid = found
            if valid is not None:
                owner, keys = owner[valid], keys[valid]
            owner, rows = self._index(hop.table, hop.column).expand(owner, keys)
        return owner, rows, table

    def hop_keys(self, from_table: str, from_column: str, rows: Any, table: str, column: str) -> Any:
        """The keys of `from_table.from_column` at `rows` (none NULL) as identities of `table.column`
        (`test.kumo_pytorch.reference_engine.gpu.data.keys`): (CuPy int64 identities, CuPy bool: which can match, or None for all). An integer,
        boolean, date or timestamp is itself, a float its bits, a text, bytes or decimal its place among the
        other column's distinct values (worked out once per pair of columns); a number of another type is
        converted into the other column's type exactly (`test.kumo_pytorch.reference_engine.gpu.data.keys.Translation`), and one with no equal
        value there (1.5 for an integer key) can match none. None when no key can match: keys of types Python
        never finds equal (an integer and a text, a date and a timestamp)."""

        from test.kumo_pytorch.reference_engine.gpu.data.keys import (
            Translation,
            key_space,
            never_equal,
        )

        source, destination = key_space(self.tables[from_table], from_column), key_space(self.tables[table], column)
        if never_equal(source.dtype, destination.dtype):
            return None
        pair = (from_table, from_column, table, column)
        if pair not in self._translations:
            self._translations[pair] = Translation(source, destination)
        return self._translations[pair].keys(rows)

    def _source(self, source: Any, call: _Call) -> _TableRows | _GroupRows:
        from kumo.pytorch.contract.sample import Rows

        key = repr(source)
        if key not in call.sources:
            call.sources[key] = (self._select(source, call) if isinstance(source, Rows)
                                 else self._group(source, self._source(source.source, call), call))
        return call.sources[key]

    def _select(self, spec: Any, call: _Call) -> _TableRows:
        """The rows `spec` selects for each anchor, grouped by anchor in anchor order."""

        import cupy

        from kumo.pytorch.contract.schema import table_clock

        owner, rows, table = self._follow(spec.path, call.table, call.rows)
        if spec.distinct and len(spec.path) > 1:
            pairs = cupy.unique(owner * self.tables[table].num_rows + rows)
            owner, rows = pairs // self.tables[table].num_rows, pairs % self.tables[table].num_rows
        target = self.tables[table]
        for condition in spec.filters:
            keep = _filter(target, condition, rows)
            owner, rows = owner[keep], rows[keep]
        clock = spec.time.column if spec.time is not None else table_clock(self.specs[table])
        if clock is not None:
            times = self._times(target, clock, rows, call.flags)
            cutoff = call.cutoffs[owner]
            keep = cupy.ones(rows.size, dtype=bool)
            if spec.time is None or spec.time.stop is not None:
                keep &= times <= cutoff + (0 if spec.time is None else spec.time.stop)
            if spec.time is not None and spec.time.start is not None:
                keep &= times > cutoff + spec.time.start
            owner, rows = owner[keep], rows[keep]
        if spec.active is not None:
            begins = self._times(target, spec.active.start_column, rows, call.flags)
            end = spec.active.end_column
            ends = _seconds(target.dtypes[end], target.values(end)[rows])
            end_present = target.present(end)
            cutoff = call.cutoffs[owner]
            open_ended = cupy.zeros(rows.size, dtype=bool) if end_present is None else ~end_present[rows]
            keep = (begins < cutoff) & (open_ended | (ends > cutoff))
            owner, rows = owner[keep], rows[keep]
        # Already grouped by anchor: each hop expands its pairs in their own order, the masks only drop
        # pairs, and `distinct` sorts by anchor first. libcudf's sorted groupby relies on that.
        return _TableRows(target, self.specs[table].primary_key, owner, rows)

    def select_rows(self, spec: Any, entity_rows: Any, cutoffs: Any, flags: list) -> _TableRows:
        """The rows a `Rows` spec selects for each anchor (entity rows and cutoffs given), grouped by anchor."""

        return self._select(spec, _Call(self.contract.sample.entity_values, self.entity, entity_rows, cutoffs, flags))

    def _times(self, table: GpuTable, column: str, rows: Any, flags: list) -> Any:
        """Clock `column` at `rows` in whole seconds; a NULL among them is refused."""

        present = table.present(column)
        if present is not None:
            flags.append((f"{table.name!r} row has no timestamp in {column!r}", (~present[rows]).any()))
        return _seconds(table.dtypes[column], table.values(column)[rows])

    def _group(self, spec: Any, inner: _TableRows | _GroupRows, call: _Call) -> _GroupRows:
        """`GroupBy`: one row per anchor and distinct key combination (NULL is a key), with its reductions."""

        import cupy

        size = inner.owner.size
        keys = [inner.column(name) for name in spec.keys]
        for name, key in zip(spec.keys, keys, strict=True):
            if key.values is None and key.words is None:
                raise ValueError(f"test.kumo_pytorch.reference_engine.gpu does not group by {key.dtype.name} values yet ({name})")
        identities = [self._identity(key, inner.owner) for key in keys]
        parts = [inner.owner]
        for key, identity in zip(keys, identities, strict=True):
            parts += [key.valid.astype(cupy.int64), cupy.where(key.valid, identity, 0)]
        order = cupy.lexsort(cupy.stack(parts[::-1])) if size else cupy.zeros(0, dtype=cupy.int64)
        rows = inner.take(order)
        keys = [key.take(order) for key in keys]
        identities = [identity[order] for identity in identities]
        change = cupy.ones(size, dtype=bool)
        if size:
            change[1:] = rows.owner[1:] != rows.owner[:-1]
            for key, identity in zip(keys, identities, strict=True):
                change[1:] |= (key.valid[1:] != key.valid[:-1]) | (key.valid[1:] & (identity[1:] != identity[:-1]))
        starts = cupy.flatnonzero(change)
        group = cupy.cumsum(change, dtype=cupy.int64) - 1
        owner = rows.owner[starts]
        group_keys = [_Column(key.dtype, None if key.values is None else key.values[starts], key.valid[starts],
                              key.texts, key.stored, None, None if key.words is None else key.words[starts],
                              None if key.numbers is None else key.numbers[starts]) for key in keys]
        for key, named in zip(keys, group_keys, strict=True):
            if key.values is not None and key.values.dtype.kind == "f":
                self._name_zeros(key, rows.owner, group, named)
        columns = {}
        for name, key in zip(spec.keys, group_keys, strict=True):
            columns[name] = _Column(key.dtype, key.values, key.valid, key.texts, key.stored, lambda: group_keys,
                                    key.words, key.numbers)
        cutoffs = call.cutoffs[owner]
        for name, reduction in spec.values.items():
            column = None if reduction.operation == "count" else rows.column(reduction.column)
            result = self._reduce(f"group.{name}", reduction.operation, column, reduction.dtype, group,
                                  int(starts.size), cutoffs, call.flags)
            stored = column.stored if reduction.operation in ("min", "max", "mode") else None
            columns[name] = _Column(result.dtype, result.values, result.valid(), result.texts, stored,
                                    lambda: group_keys,  # upstream orders groups by their keys
                                    result.words, result.numbers)
        return _GroupRows(owner, columns)

    def _identity(self, key: _Column, owner: Any) -> Any:
        """Integers equal exactly when upstream's grouping takes two of `key`'s values for one key (a Python dict's
        keys): by value; a decimal by its place among the distinct values; a float by its bits, -0.0 as 0.0
        (they are equal), and every NaN apart (a dict holds NaNs by object, and NaN equals nothing): its own NaN
        bits, numbered in upstream's row order, so NaN groups keep upstream's order among themselves."""

        import cupy

        if key.words is not None:
            from test.kumo_pytorch.reference_engine.gpu.data.decimals import (
                distinct_index,
            )

            return distinct_index(key.words)[0]
        if key.values.dtype.kind != "f":
            return _bits(key.values)
        from test.kumo_pytorch.reference_engine.gpu.data.keys import widened

        numbers = widened(key.values)  # a float32 exactly (CuPy's own widening flushes subnormals)
        identity = cupy.where(numbers == 0, cupy.int64(0), numbers.view(cupy.int64))
        nan = cupy.isnan(numbers) & key.valid
        if bool(nan.any()):  # a flag read; the rest only where a NaN is
            rows = cupy.flatnonzero(nan)
            identity[rows] = cupy.int64(0x7FF8 << 48) | self._upstream_ranks(key, owner, rows)
        return identity

    def _upstream_ranks(self, column: _Column, owner: Any, rows: Any) -> Any:
        """The place of each of `rows` among them in upstream's row order (CuPy int64): by anchor, then by the
        printed sort keys of the rows (a table's primary key, a grouping's keys), ties as given. Worked out on
        the host, for these rows only."""

        from test.kumo_pytorch.reference_engine.gpu.data.printed import (
            sort_key,
        )

        import cupy

        keys = [_column_python(key.take(rows)) for key in column.sort_keys()]
        owners = owner[rows].get().tolist()
        order = sorted(range(len(owners)), key=lambda i: (owners[i], tuple(sort_key(key[i]) for key in keys)))
        ranks = cupy.empty(len(owners), dtype=cupy.int64)
        ranks[cupy.asarray(order, dtype=cupy.int64)] = cupy.arange(len(owners), dtype=cupy.int64)
        return ranks

    def _name_zeros(self, key: _Column, owner: Any, group: Any, named: _Column) -> None:
        """A group holding both -0.0 and 0.0 keys is named, as upstream's dict names it, by its first row in
        upstream's order: its key is that row's zero. Worked out only where a group holds both."""

        import cupy

        zero = key.valid & (key.values == 0)
        count = int(named.valid.size)
        negative = _counts(group[zero & cupy.signbit(key.values)], count)
        positive = _counts(group[zero & ~cupy.signbit(key.values)], count)
        mixed = (negative > 0) & (positive > 0)
        if not bool(mixed.any()):
            return
        rows = cupy.flatnonzero(zero & mixed[group])
        ranks = self._upstream_ranks(key, owner, rows)
        first = cupy.full(count, cupy.iinfo(cupy.int64).max, dtype=cupy.int64)
        cupy.minimum.at(first, group[rows], ranks)
        chosen = rows[ranks == first[group[rows]]]
        named.values[group[chosen]] = key.values[chosen]

    # -- reductions -------------------------------------------------------------------------------

    def _reduce(self, where: str, operation: str, column: _Column | None, dtype: Any, segment: Any, count: int,
                cutoffs: Any, flags: list) -> Values:
        """One reduction of `column` per segment (segments sorted, `count` of them; `cutoffs` per segment)."""

        import cupy

        rows_per = _counts(segment, count)
        if operation == "count":
            return Values(dtype, rows_per)
        if column.dtype.name.startswith("decimal"):
            if operation in ("mean", "var_population", "var_sample", "std_population", "std_sample"):
                from kumo.pytorch.contract.dtypes import DType

                column = _Column(DType("float64"), column.numbers, column.valid)  # upstream: float(value)
            else:
                return self._decimal_reduce(where, operation, column, dtype, segment, rows_per, count, flags)
        if column.values is None:
            raise ValueError(f"{where}: test.kumo_pytorch.reference_engine.gpu does not reduce {column.dtype.name} columns yet")
        valid, values = column.valid, column.values
        if operation == "count_distinct":
            return Values(dtype, _count_distinct(segment[valid], _bits(values[valid]), count))
        if operation == "sum" and column.dtype.integer:
            return _exact_sum(where, segment[valid], values[valid], rows_per, dtype, flags)
        if operation == "mode":
            return self._mode(column, segment[valid], values[valid], count, dtype)
        if operation == "recency":
            seconds = _seconds(column.dtype, values)
            latest, has = _grouped(segment[valid], seconds[valid], "max", count)
            return Values(dtype, cupy.where(has, (cutoffs - latest).astype(cupy.float64), 0.0), has)
        if operation in ("min", "max") and values.dtype.kind == "f":
            return self._extreme(operation, column, segment, count, dtype)
        if operation in ("min", "max"):
            result, has = _grouped(segment[valid], values[valid], operation, count)
            return Values(dtype, result, has, column.texts)
        numeric = values.astype(cupy.float64)
        # Float sums and means: upstream adds from +0.0, so a sum of zeros is never -0.0 (adding +0.0 makes it so)
        if operation == "sum":  # an empty selection sums to 0.0, all-NULL values to NULL
            total, has = _grouped(segment[valid], numeric[valid], "sum", count)
            empty = rows_per == 0
            return Values(dtype, cupy.where(empty, 0.0, total) + 0.0, has | empty)
        if operation == "mean":
            result, has = _grouped(segment[valid], numeric[valid], "mean", count)
            return Values(dtype, result + 0.0, has)
        if operation in ("var_population", "var_sample", "std_population", "std_sample"):
            ddof = 1 if operation.endswith("sample") else 0
            kind = "variance" if operation.startswith("var") else "std"
            result, has = _grouped(segment[valid], numeric[valid], kind, count, ddof=ddof)
            return Values(dtype, result, has)
        raise ValueError(f"{where}: test.kumo_pytorch.reference_engine.gpu does not build the {operation!r} reduction yet")

    def _decimal_reduce(self, where: str, operation: str, column: _Column, dtype: Any, segment: Any, rows_per: Any,
                        count: int, flags: list) -> Values:
        """A reduction of decimals per segment, as upstream reduces Python ``Decimal``: an exact sum (refused, with
        upstream's message, past the result's precision); the extremes and distinct count of their values; the
        most frequent value, ties to the smallest printed value."""

        from test.kumo_pytorch.reference_engine.gpu.data.decimals import (
            calculate,
            decimal_values,
            distinct_index,
            segment_sums,
        )

        import cupy

        valid = column.valid
        owner, held = segment[valid], column.words[valid]
        if operation == "sum":  # an empty selection sums to 0, all-NULL values to NULL
            limbs, total = segment_sums(owner, held, count)
            present = (rows_per == 0) | (_counts(owner, count) > 0)
            sign = (limbs[:, 7] >> 31) & 1
            over = limbs[:, 8] != sign * (2**32 - 1)  # past 256 bits
            _, outside = calculate("abs", [Values(dtype, None, words=total)], dtype)
            bad = present & (over | outside)
            flags.append((lambda: _sum_refusal(where, dtype, limbs, bad), bad.any()))
            return decimal_values(dtype, total, present)
        index, distinct = distinct_index(held)
        if operation == "count_distinct":
            return Values(dtype, _count_distinct(owner, index, count))
        if operation in ("min", "max"):
            place, has = _grouped(owner, index, operation, count)
        elif operation == "mode":
            ranks = _printed_ranks(distinct, column.dtype)  # upstream's tie order: (type name, repr)
            best, has = _most_frequent(owner, ranks[index], count)
            place = cupy.argsort(ranks)[best]
        else:
            raise ValueError(f"{where}: test.kumo_pytorch.reference_engine.gpu does not build the {operation!r} reduction of decimals yet")
        if distinct.shape[0] == 0:
            return decimal_values(dtype, cupy.zeros((count, 4), dtype=cupy.int64), has)
        return decimal_values(dtype, distinct[cupy.where(has, place, 0)], has)

    def _extreme(self, operation: str, column: _Column, segment: Any, count: int, dtype: Any) -> Values:
        """Float `min`/`max` as upstream's Python `min`/`max` over each segment's values in its row order: NaN
        when the first value is NaN (later NaNs are skipped), else the first value equal to the extreme, so
        -0.0 or 0.0, whichever comes first. Segments where that order matters (a NaN, or both zeros at the
        extreme) are settled on the host; the others need only the GPU's extreme of their non-NaN values."""

        import cupy

        valid = column.valid
        segments, values = segment[valid], column.values[valid]
        nan = cupy.isnan(values)
        result, _ = _grouped(segments[~nan], values[~nan], operation, count)
        zero = ~nan & (values == 0) & (result[segments] == 0)
        negative = _counts(segments[zero & cupy.signbit(values)], count)
        positive = _counts(segments[zero & ~cupy.signbit(values)], count)
        unsettled = (_counts(segments[nan], count) > 0) | ((negative > 0) & (positive > 0))
        if bool(unsettled.any()):
            picked = unsettled[segments]
            rows = cupy.flatnonzero(valid)[picked]
            self._settle(operation, column, rows, segments[picked], values[picked], result)
        return Values(dtype, result, _counts(segments, count) > 0, column.texts)

    def _settle(self, operation: str, column: _Column, rows: Any, segments: Any, values: Any, result: Any) -> None:
        """Python's `min`/`max` over the given rows' values, per segment, in upstream's row order."""

        from collections import defaultdict
        from test.kumo_pytorch.reference_engine.gpu.data.printed import (
            sort_key,
        )

        import cupy

        keys = [_column_python(key.take(rows)) for key in column.sort_keys()]
        by_segment = defaultdict(list)
        for index, (segment, value) in enumerate(zip(segments.get().tolist(), values.get().tolist(), strict=True)):
            by_segment[segment].append((tuple(sort_key(key[index]) for key in keys), value))
        choose = min if operation == "min" else max
        settled = {segment: choose(value for _, value in sorted(items, key=lambda item: item[0]))
                   for segment, items in by_segment.items()}
        result[cupy.asarray(list(settled), dtype=cupy.int64)] = cupy.asarray(list(settled.values()),
                                                                             dtype=result.dtype)

    def _mode(self, column: _Column, owner: Any, values: Any, count: int, dtype: Any) -> Values:
        """The most frequent value per segment; ties to the smallest (type name, printed value), as upstream."""

        import cupy

        keys = _bits(values)
        order = cupy.lexsort(cupy.stack([keys, owner]))
        owner, keys, values = owner[order], keys[order], values[order]
        if owner.size == 0:
            return Values(dtype, cupy.zeros(count, dtype=values.dtype), cupy.zeros(count, dtype=bool), column.texts)
        starts = cupy.flatnonzero(cupy.concatenate([cupy.ones(1, dtype=bool),
                                                    (owner[1:] != owner[:-1]) | (keys[1:] != keys[:-1])]))
        lengths = cupy.diff(cupy.concatenate([starts, cupy.asarray([owner.size])]))
        run_owner, run_value = owner[starts], values[starts]
        most = cupy.zeros(count, dtype=cupy.int64)
        cupy.maximum.at(most, run_owner, lengths)
        tied = lengths == most[run_owner]
        distinct_keys, rank_of_key, values_by_rank = self._rank(column)
        rank = rank_of_key[cupy.searchsorted(distinct_keys, _bits(run_value))]
        best = cupy.full(count, cupy.iinfo(cupy.int64).max, dtype=cupy.int64)
        cupy.minimum.at(best, run_owner[tied], rank[tied])
        has = most > 0
        return Values(dtype, values_by_rank[cupy.where(has, best, 0)], has, column.texts)

    def _rank(self, column: _Column) -> tuple[Any, Any, Any]:
        """Each distinct value's place in upstream's tie order, (type name, repr) of the Python value: built
        once over a stored column's values, or over a calculated column's own."""

        import cupy

        if column.stored in self._ranks:
            return self._ranks[column.stored]
        if column.stored is not None:
            table = self.tables[column.stored[0]]
            present = table.present(column.stored[1])
            stored = table.values(column.stored[1])
            stored = stored if present is None else stored[present]
        else:
            stored = column.values[column.valid]
        distinct_keys, first = cupy.unique(_bits(stored), return_index=True)
        distinct_values = stored[first]
        python = _python(column.dtype, column.texts, distinct_values)
        order = sorted(range(len(python)), key=lambda i: (type(python[i]).__name__, repr(python[i])))
        order = cupy.asarray(order, dtype=cupy.int64)
        rank_of_key = cupy.empty(order.size, dtype=cupy.int64)
        rank_of_key[order] = cupy.arange(order.size, dtype=cupy.int64)
        result = (distinct_keys, rank_of_key, distinct_values[order])
        if column.stored is not None:
            self._ranks[column.stored] = result
        return result
