# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Event sequences, computed on the GPU for all of a call's anchors at once.

Kumo PyTorch's G-HISTORY-001 rules:

- Each event component (``SampleSpec.events``, or ``current_events``) selects
  its rows for each anchor as an aggregate does (path, filters, time window,
  active interval; a selected row with no time is refused), orders them by time
  and then by printed primary key (upstream's row order), and keeps ``limit`` of
  them from the end its ``order`` names. Each event's fields are worked out with
  the event as the base row: its own columns, fields through relations (as of
  the anchor's cutoff, or of the event's own time), constants, calculations, and
  ``EventTimeDelta`` (the anchor's cutoff minus the event's time, in its unit).
- An ``EventSource`` input merges the components it names, oldest first (ties
  in the order it names them, then each component's own order), keeps
  ``max_events`` from the end ``keep`` names, and delivers them in ``order``. A
  field it reads is NULL on events of a component that does not declare it.
- A ``Tokens`` input with ``missing="skip_event"`` then drops each event where a
  field it encodes, declared by the event's component, is NULL; a ``Composite``
  input each event where one of its ``required_fields`` is NULL (upstream's
  ``_eligible``). A Composite input's encoders are fitted on every event, so its
  events before that are kept for fitting.
- An ``EventsOf`` input reads the very events its parent keeps, in its order.
- The fields an ``AlignedEventTarget`` withholds from an input are read with
  the input's own (`test.kumo_pytorch.reference_engine.gpu.encoding.targets.aligned_reads`).

Integer primary keys are ordered by their printed form on the GPU; other keys
are ordered by value, and only events of one anchor at the same second are put
in printed-key order on the host.
"""

from __future__ import annotations

from dataclasses import dataclass
from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
    Values,
    recoded,
    shared_texts,
)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from test.kumo_pytorch.reference_engine.gpu.sampling.sampler import Sampler

    from kumo.pytorch.spec import DataLoaderSpec

__all__ = ["EventRows", "event_inputs"]


@dataclass
class EventRows:
    """An event input's events for a call's anchors, in delivery order and grouped by anchor.

    :param lengths: events per anchor (CuPy int64).
    :param values: each field the input (or an ``EventsOf`` follower of it) reads, one value per event.
    :param declared: for each of those fields, whether each event's component declares it (a CuPy bool
        array), or None where every component the input names does.
    :param fitting: a ``Composite`` input's values before its required fields dropped events (its encoders
        are fitted on those), or None when that drops none.
    :param components: each event's component, by its place in the source's ``events`` (CuPy int64): what a
        field's identity, and so its field-name ID, is on that event.
    """

    lengths: Any
    values: dict[str, Values]
    declared: dict[str, Any]
    fitting: dict[str, Values] | None = None
    components: Any = None

    def select(self, keep: Any) -> EventRows:
        """Only the events where `keep` (a CuPy bool array) holds, in the same order."""

        from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
            _counts,
        )

        import cupy

        owner = cupy.searchsorted(cupy.cumsum(self.lengths), cupy.arange(keep.size, dtype=cupy.int64), side="right")
        values = {name: Values(value.dtype, value.values[keep], value.valid()[keep], value.texts)
                  for name, value in self.values.items()}
        declared = {name: None if mask is None else mask[keep] for name, mask in self.declared.items()}
        return EventRows(_counts(owner[keep], self.lengths.size), values, declared,
                         components=None if self.components is None else self.components[keep])


@dataclass
class _Component:
    """One component's events for a call's anchors: grouped by anchor, oldest first (printed key on ties)."""

    spec: Any
    table: str
    owner: Any
    rows: Any
    times: Any
    rank: Any  # place among the anchor's events of this component, oldest first


def event_inputs(contract: DataLoaderSpec, sampler: Sampler, entity_rows: Any, cutoffs: Any,
                 flags: list[tuple[Any, Any]]) -> dict[str, EventRows]:
    """Every event input's events and read fields, for the anchors whose entity rows and cutoffs are given."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.targets import (
        aligned_reads,
    )

    import cupy

    from kumo.pytorch.contract.batch import Composite, EventsOf, EventSource
    from kumo.pytorch.contract.batch.validation import resolved_fields

    inputs = contract.batch.inputs
    withheld = aligned_reads(contract)
    reads = {name: tuple(dict.fromkeys((*(use.field for use in getattr(value.representation, "fields", ())),
                                        *withheld.get(name, ())))) for name, value in inputs.items()}
    readers: dict[str, dict[str, Any]] = {}  # parent EventSource input -> selector -> upstream's resolution
    for name, value in inputs.items():
        parent = value.source.input if isinstance(value.source, EventsOf) else name
        if isinstance(inputs[parent].source, EventSource):
            selected = readers.setdefault(parent, {})
            for selector in reads[name]:
                selected[selector] = resolved_fields(contract, name, selector)
    needed: dict[tuple[str, str], set[str]] = {}
    for parent, selected in readers.items():
        source = inputs[parent].source
        collection = "events" if source.collection == "history" else "current_events"
        for component in source.events:
            needed.setdefault((collection, component), set())
        for variants in selected.values():
            for identity, _, cardinality in variants:
                if cardinality != "scalar":
                    raise ValueError(f"test.kumo_pytorch.reference_engine.gpu does not build repeated event fields yet ({'/'.join(identity)})")
                needed[(identity[0], identity[1])].add(identity[2])
    count = entity_rows.size
    components, fields = {}, {}
    for (collection, name), names in needed.items():
        spec = getattr(contract.sample, collection)[name]
        component = _component(sampler, spec, entity_rows, cutoffs, count, flags)
        components[(collection, name)] = component
        fields[(collection, name)] = sampler.scope_values(sorted(names), spec.fields, component.table, component.rows,
                                                          cutoffs[component.owner], component.times, flags)
    result = {}
    for parent, selected in readers.items():
        source = inputs[parent].source
        collection = "events" if source.collection == "history" else "current_events"
        parts = [components[(collection, name)] for name in source.events]
        lengths, which, local = _merge(parts, source, count)
        values, declared = {}, {}
        for selector, variants in selected.items():
            values[selector] = _gathered(selector, variants, source.events, fields, collection, which, local)
            owners = {source.events.index(identity[1]) for identity, _, _ in variants}
            declared[selector] = None if len(owners) == len(source.events) else \
                cupy.isin(which, cupy.asarray(sorted(owners), dtype=cupy.int64))
        rows = EventRows(lengths, values, declared, components=which)
        representation = inputs[parent].representation
        if isinstance(representation.row, Composite) and representation.row.required_fields:
            kept = rows.select(_required(rows.values, representation.row.required_fields))
            rows = EventRows(kept.lengths, kept.values, kept.declared, fitting=rows.values, components=kept.components)
        elif getattr(representation.row, "missing", None) == "skip_event":
            rows = rows.select(_complete(rows, [use.field for use in representation.fields]))
        result[parent] = rows
    for name, value in inputs.items():
        if isinstance(value.source, EventsOf) and value.source.input in result:
            parent = result[value.source.input]
            result[name] = EventRows(parent.lengths, {field: parent.values[field] for field in reads[name]},
                                     {field: parent.declared[field] for field in reads[name]},
                                     components=parent.components)
    return result


def _required(values: dict[str, Values], fields: tuple[str, ...]) -> Any:
    """Rows where none of `fields` is NULL (a ``Composite`` input's required fields), declared or not."""

    import cupy

    keep = cupy.ones(values[fields[0]].valid().size, dtype=bool)
    for name in fields:
        keep &= values[name].valid()
    return keep


def _complete(rows: EventRows, fields: list[str]) -> Any:
    """Events where none of `fields` is NULL on a component that declares it (``skip_event``)."""

    import cupy

    keep = cupy.ones(int(rows.lengths.sum()), dtype=bool)
    for name in fields:
        valid = rows.values[name].valid()
        declared = rows.declared[name]
        keep &= valid if declared is None else ~declared | valid
    return keep


def _component(sampler: Sampler, spec: Any, entity_rows: Any, cutoffs: Any, count: int, flags: list) -> _Component:
    """One component's events for each anchor: selected, ordered oldest first, `limit` kept."""

    from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
        _counts,
    )

    from kumo.pytorch.contract.schema import table_clock

    selected = sampler.select_rows(spec.rows, entity_rows, cutoffs, flags)
    table = selected.table.name
    clock = spec.rows.time.column if spec.rows.time is not None else table_clock(sampler.specs[table])
    owner, rows = selected.owner, selected.rows
    times = sampler._times(selected.table, clock, rows, flags)
    order = _chronological(sampler, table, owner, rows, times)
    owner, rows, times = owner[order], rows[order], times[order]
    rank = _ranks(owner)
    if spec.limit is not None:
        if spec.order == "ascending":
            keep = rank < spec.limit
        else:
            keep = rank >= _counts(owner, count)[owner] - spec.limit
        owner, rows, times, rank = owner[keep], rows[keep], times[keep], rank[keep]
    return _Component(spec, table, owner, rows, times, rank)


def _chronological(sampler: Sampler, table_name: str, owner: Any, rows: Any, times: Any) -> Any:
    """The order of (anchor, time, printed primary key): upstream's event order within an anchor."""

    from test.kumo_pytorch.reference_engine.gpu.data.keys import identities
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

    table = sampler.tables[table_name]
    key = sampler.specs[table_name].primary_key
    keys = identities(table, key)[0][rows]  # one integer a key value: ties are put in printed order below
    if on_gpu(table.dtypes[key]):  # printed order on the GPU
        columns = [owner, cupy.ascontiguousarray(times), *key_columns(table, key, rows)]
        return _view(plc.sorting.stable_sorted_order(plc.Table([_column(column) for column in columns]),
                                                     [plc.types.Order.ASCENDING] * len(columns),
                                                     [plc.types.NullOrder.AFTER] * len(columns))).astype(cupy.int64)
    order = cupy.lexsort(cupy.stack([keys, times, owner]))
    if order.size < 2:
        return order
    tied_owner, tied_time = owner[order], times[order]
    tied = (tied_owner[1:] == tied_owner[:-1]) & (tied_time[1:] == tied_time[:-1])
    if bool(tied.any()):
        order = _settle_ties(sampler, table_name, key, rows, order, tied)
    return order


def _settle_ties(sampler: Sampler, table_name: str, key: str, rows: Any, order: Any, tied: Any) -> Any:
    """Runs of one anchor's events at the same second, put in printed-key order on the host (floats, decimals,
    bytes: `test.kumo_pytorch.reference_engine.gpu.data.printed.on_gpu`)."""

    from test.kumo_pytorch.reference_engine.gpu.data.printed import sort_key
    from test.kumo_pytorch.reference_engine.gpu.sampling.host_values import (
        _values_python,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
        _column_values,
    )

    import cupy

    table = sampler.tables[table_name]
    member = cupy.zeros(order.size, dtype=bool)
    member[1:] |= tied
    member[:-1] |= tied
    positions = cupy.flatnonzero(member)
    starts = cupy.ones(positions.size, dtype=bool)
    starts[1:] = ~tied[positions[1:] - 1]
    run = cupy.cumsum(starts) - 1
    picked = order[positions]
    printed = _values_python(_column_values(table, key, rows[picked]))
    groups: dict[int, list[int]] = {}
    for index, run_id in enumerate(run.get().tolist()):
        groups.setdefault(run_id, []).append(index)
    settled = list(range(positions.size))
    for members in groups.values():
        ordered = sorted(members, key=lambda index: sort_key(printed[index]))
        for slot, index in zip(sorted(members), ordered, strict=True):
            settled[slot] = index
    result = order.copy()
    result[positions] = picked[cupy.asarray(settled, dtype=cupy.int64)]
    return result


def _ranks(owner: Any) -> Any:
    """Each entry's place among its owner's entries (owners grouped, in order)."""

    import cupy

    return cupy.arange(owner.size, dtype=cupy.int64) - cupy.searchsorted(owner, owner, side="left")


def _merge(parts: list[_Component], source: Any, count: int) -> tuple[Any, Any, Any]:
    """An `EventSource`'s events per anchor: (events per anchor, component of each, its index there)."""

    from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
        _counts,
    )

    import cupy

    owner = cupy.concatenate([part.owner for part in parts])
    times = cupy.concatenate([part.times for part in parts]).astype(cupy.int64)
    which = cupy.concatenate([cupy.full(part.owner.size, index, dtype=cupy.int64)
                              for index, part in enumerate(parts)])
    rank = cupy.concatenate([part.rank for part in parts])
    local = cupy.concatenate([cupy.arange(part.owner.size, dtype=cupy.int64) for part in parts])
    order = cupy.lexsort(cupy.stack([rank, which, times, owner])) if owner.size else owner
    owner, which, local = owner[order], which[order], local[order]
    if source.max_events is not None:
        position = _ranks(owner)
        if source.keep == "last":
            keep = position >= _counts(owner, count)[owner] - source.max_events
        else:
            keep = position < source.max_events
        owner, which, local = owner[keep], which[keep], local[keep]
    lengths = _counts(owner, count)
    if source.order == "descending":
        start = cupy.cumsum(lengths) - lengths
        position = _ranks(owner)
        reverse = cupy.empty_like(position)
        reverse[start[owner] + lengths[owner] - 1 - position] = cupy.arange(owner.size, dtype=cupy.int64)
        which, local = which[reverse], local[reverse]
    return lengths, which, local


def _merged_arrow(dtype: Any, parts: list, which: Any, local: Any) -> Any:
    """A decimal event field's values (Arrow, NULLs not yet applied): each event's from its own component, on the
    host (asked for only by a Raw output, which is an Arrow array itself)."""

    import numpy as np
    import pyarrow as pa

    arrays = [part.arrow() for _, part in parts]
    total = sum(len(array) for array in arrays)
    if total == 0:
        return pa.nulls(int(which.size), dtype.to_arrow())
    starts = np.zeros(max(index for index, _ in parts) + 1, dtype=np.int64)
    declared = np.zeros(starts.size, dtype=bool)
    offset = 0
    for (index, _), array in zip(parts, arrays, strict=True):
        starts[index], declared[index] = offset, True
        offset += len(array)
    which, local = which.get(), local.get()
    hit = (which < starts.size) & declared[np.minimum(which, starts.size - 1)]
    at = np.where(hit, starts[np.minimum(which, starts.size - 1)] + local, 0)
    return pa.concat_arrays([array.cast(dtype.to_arrow()) for array in arrays]).take(pa.array(at))


def _gathered(selector: str, variants: Any, components: tuple[str, ...], fields: dict, collection: str,
              which: Any, local: Any) -> Values:
    """One read field over an input's events: each event's value from its own component (NULL where the
    component does not declare the field), text codes put into one dictionary."""

    from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
        _device_type,
    )

    import cupy

    dtypes = {dtype for _, dtype, _ in variants}
    if len(dtypes) != 1:
        raise ValueError(f"test.kumo_pytorch.reference_engine.gpu does not merge event field {selector!r} of different types yet")
    dtype = dtypes.pop()
    parts = []
    for identity, _, _ in variants:
        part = fields[(collection, identity[1])][identity[2]]
        if part.values is None and part.numbers is None:
            raise ValueError(f"test.kumo_pytorch.reference_engine.gpu does not build {dtype.name} event fields yet ({selector})")
        parts.append((components.index(identity[1]), part))
    if dtype.name.startswith("decimal"):  # each event's number from its own component; the values when asked
        numbers = cupy.zeros(which.size, dtype=cupy.float64)
        present = cupy.zeros(which.size, dtype=bool)
        for index, part in parts:
            if part.numbers.size == 0:
                continue
            hit = which == index
            at = cupy.where(hit, local, 0)
            numbers = cupy.where(hit, part.numbers[at], numbers)
            present = cupy.where(hit, part.valid()[at], present)
        return Values(dtype, None, present, numbers=numbers, build=lambda: _merged_arrow(dtype, parts, which, local))
    texts, codes = shared_texts([part for _, part in parts])
    values = cupy.zeros(which.size, dtype=_device_type(dtype))
    present = cupy.zeros(which.size, dtype=bool)
    for (index, part), remap in zip(parts, codes, strict=True):
        if part.values.size == 0:
            continue
        hit = which == index
        at = cupy.where(hit, local, 0)
        picked = recoded(part.values[at], remap)
        values = cupy.where(hit, picked, values)
        present = cupy.where(hit, part.valid()[at], present)
    return Values(dtype, values, present, texts)
