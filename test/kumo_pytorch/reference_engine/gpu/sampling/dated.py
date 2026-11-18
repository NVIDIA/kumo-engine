# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dated entity values (``DatedEntityValue``), sampled on the GPU for all of a call's anchors at once.

The rules follow Kumo PyTorch's G-FIELDS-001 contract. For each anchor, a dated value selects rows as
an aggregate does (``rows``: a path of at most one relation here, a time window
on the destination's clock, filters) and reads one value from each (``field``:
a column of the row itself, or a ``Constant``), dated by the row's clock:

- a row whose value is NULL gives no occurrence (a NULL ``Constant`` does: one
  per row, its value missing, as upstream reads a constant whatever it is);
- ``first_per_value`` keeps each distinct value's earliest occurrence (equal
  times: the row upstream reads first, by printed primary key);
- every occurrence gets its occurrence ``fields`` (like event fields: its time
  delta, the row's columns, constants, calculations over them and the value);
- an anchor's occurrences are in upstream's order: by time, then by value as
  upstream prints it (``repr``: ``10`` before ``9``), then by
  printed primary key.

An entity input reads a dated value as a list per anchor (`test.kumo_pytorch.reference_engine.gpu.batching`).
Values of integers, text, floats and booleans are built.
"""

from __future__ import annotations

from dataclasses import dataclass
from test.kumo_pytorch.reference_engine.gpu.data.text import python_texts
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
    from test.kumo_pytorch.reference_engine.gpu.sampling.sampler import Sampler

__all__ = ["Occurrences", "dated_values"]


@dataclass
class Occurrences:
    """One dated value's occurrences for a call's anchors, anchor by anchor in upstream's order.

    :param owner: each occurrence's anchor position in the call (CuPy int64).
    :param values: the value of each (`test.kumo_pytorch.reference_engine.gpu.encoding.encoders.Values`).
    :param times: each one's time, in seconds (CuPy int64).
    :param fields: each occurrence field's values, by name.
    :param bounds: where each anchor's occurrences start, then the end (host int64).
    """

    owner: Any
    values: Values
    times: Any
    fields: dict[str, Values]
    bounds: Any


def dated_values(sampler: Sampler, names: list[str], entity_rows: Any, cutoffs: Any,
                 flags: list[tuple[Any, Any]]) -> dict[str, Occurrences]:
    """Each named dated value's occurrences for the anchors whose entity rows and cutoffs are given."""

    return {name: _dated(sampler, name, sampler.contract.sample.dated_entity_values[name], entity_rows, cutoffs,
                         flags) for name in names}


def _dated(sampler: Sampler, name: str, spec: Any, entity_rows: Any, cutoffs: Any,
           flags: list[tuple[Any, Any]]) -> Occurrences:
    from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
        _column_values,
    )

    import cupy

    from kumo.pytorch.contract.sample import Constant
    from kumo.pytorch.contract.schema import table_clock

    count = int(entity_rows.size)
    selected = sampler.select_rows(spec.rows, entity_rows, cutoffs, flags)
    table, owner, rows = selected.table, selected.owner, selected.rows
    clock = spec.rows.time.column if spec.rows.time is not None else table_clock(sampler.specs[table.name])
    times = sampler._times(table, clock, rows, flags)
    values = sampler._constant(name, spec.field, int(rows.size)) if isinstance(spec.field, Constant) else \
        _column_values(table, spec.field.column, rows)  # a constant: its value on every row selected
    if values.present is not None and not isinstance(spec.field, Constant):  # a NULL read is no occurrence
        kept = cupy.flatnonzero(values.present)
        owner, rows, times, values = owner[kept], rows[kept], times[kept], _taken(values, kept)
    printed_rank = _printed_ranks(table, sampler.specs[table.name].primary_key, rows)
    value_rank = _value_ranks(values)
    if spec.first_per_value and rows.size:  # each value's earliest occurrence, ties to the row read first
        order = cupy.lexsort(cupy.stack([printed_rank, times, value_rank, owner]))
        first = cupy.ones(order.size, dtype=bool)
        first[1:] = (owner[order][1:] != owner[order][:-1]) | (value_rank[order][1:] != value_rank[order][:-1])
        keep = cupy.sort(order[first])
        owner, rows, times, printed_rank, value_rank = (owner[keep], rows[keep], times[keep], printed_rank[keep],
                                                        value_rank[keep])
        values = _taken(values, keep)
    if rows.size:
        order = cupy.lexsort(cupy.stack([printed_rank, value_rank, times, owner]))
        owner, rows, times = owner[order], rows[order], times[order]
        values = _taken(values, order)
    scope = {name: spec.field, **spec.fields}
    fields = sampler.scope_values(list(spec.fields), scope, table.name, rows, cutoffs[owner], times, flags) \
        if spec.fields else {}
    bounds = cupy.searchsorted(owner, cupy.arange(count + 1, dtype=cupy.int64)).get()
    return Occurrences(owner, values, times, fields, bounds)


def _taken(values: Values, index: Any) -> Values:
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values

    return Values(values.dtype, values.values[index], None if values.present is None else values.present[index],
                  values.texts)


def _printed_ranks(table: Any, key: str, rows: Any) -> Any:
    """Each row's place in the order of its printed primary key (upstream reads a relation's rows so):
    `test.kumo_pytorch.reference_engine.gpu.data.printed.key_columns`, an integer's digits or a text's ``repr``."""

    from test.kumo_pytorch.reference_engine.gpu.data.printed import key_columns
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        _column,
        _view,
    )

    import cupy
    import pylibcudf as plc

    if rows.size == 0:
        return cupy.zeros(0, dtype=cupy.int64)
    columns = key_columns(table, key, rows)
    order = _view(plc.sorting.stable_sorted_order(plc.Table([_column(column) for column in columns]),
                                                  [plc.types.Order.ASCENDING] * len(columns),
                                                  [plc.types.NullOrder.AFTER] * len(columns))).astype(cupy.int64)
    rank = cupy.empty(rows.size, dtype=cupy.int64)
    rank[order] = cupy.arange(rows.size, dtype=cupy.int64)
    return rank


def _value_ranks(values: Values) -> Any:
    """Each value's place in upstream's value order (``_sort_value``: its ``repr``), equal values equal, NULL
    first (-1)."""

    import cupy

    valid = values.valid()
    size = int(valid.size)
    if size == 0:
        return cupy.zeros(0, dtype=cupy.int64)
    if values.texts is not None:  # each distinct text's repr, ranked once on the host
        texts = python_texts(values.texts, values.dtype)
        ranked = sorted(range(len(texts)), key=lambda code: repr(texts[code]))
        table = cupy.empty(max(len(texts), 1), dtype=cupy.int64)
        table[cupy.asarray(ranked, dtype=cupy.int64)] = cupy.arange(len(texts), dtype=cupy.int64)
        codes = cupy.where(valid, values.values, 0).astype(cupy.int64)
        return cupy.where(valid, table[codes], -1)
    if values.dtype.name == "bool":
        return cupy.where(valid, values.values.astype(cupy.int64), -1)
    if values.dtype.name.startswith("float"):
        keys = values.values.astype(cupy.float64).view(cupy.int64)
    elif values.values.dtype == cupy.uint64:  # unsigned as it is: 2**63 and up print as themselves, not negative
        keys = values.values
    else:
        keys = values.values.astype(cupy.int64)
    distinct, inverse = cupy.unique(cupy.where(valid, keys, keys.dtype.type(0)), return_inverse=True)
    host = distinct.get()
    if values.dtype.name.startswith("float"):
        import numpy as np

        printed = [repr(float(value)) for value in host.view(np.float64)]
    else:
        printed = [repr(int(value)) for value in host]
    order = sorted(range(len(printed)), key=lambda position: printed[position])
    rank = [0] * len(printed)
    previous, current = None, -1
    for position in order:  # equal reprs share a rank (NaNs of other bits print alike)
        if printed[position] != previous:
            current += 1
            previous = printed[position]
        rank[position] = current
    return cupy.where(valid, cupy.asarray(rank, dtype=cupy.int64)[inverse.ravel()], -1)
