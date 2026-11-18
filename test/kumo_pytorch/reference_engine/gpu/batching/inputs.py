# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A call's input values: which inputs, their configuration, and their sampled values."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
from test.kumo_pytorch.reference_engine.gpu.sampling.sampler import Sampler
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.spec import DataLoaderSpec

    from test.kumo_pytorch.reference_engine.gpu.data.anchors import GpuAnchors

from test.kumo_pytorch.reference_engine.gpu.sampling.indexes import EntityIndex


def _inputs(contract: DataLoaderSpec) -> Iterator[tuple[str, Any]]:
    for name, value in contract.batch.inputs.items():
        for use in getattr(value.representation, "fields", ()):  # graph inputs: none yet
            yield name, use


def _config(contract: DataLoaderSpec, use: Any) -> Any:
    """A field's encoding, a shared one (``UseEncoding``) looked up in ``batch.encodings``."""

    from kumo.pytorch.contract.batch.encodings import UseEncoding

    return contract.batch.encodings[use.encoding.name] if isinstance(use.encoding, UseEncoding) else use.encoding


@dataclass
class _InputValues:
    """One input's sampled values for a call: one per anchor (entity column inputs), or one per row with
    each anchor's number of rows (event inputs, and entity token or Composite inputs: their one row, or none
    if ``skip_event`` or a required field drops it).

    `declared` holds, for an event field, whether each event's component declares it (None: every one);
    `fitting`, a Composite input's values before required fields dropped rows (None: none dropped);
    `components`, an event input's component of each event (None: an entity input)."""

    lengths: Any
    values: dict[str, Values]
    declared: dict[str, Any] = dataclass_field(default_factory=dict)
    fitting: dict[str, Values] | None = None
    dated: dict[str, Any] = dataclass_field(default_factory=dict)  # a dated field's `test.kumo_pytorch.reference_engine.gpu.sampling.dated.Occurrences`
    components: Any = None


def _sampled(contract: DataLoaderSpec, sampler: Sampler, index: EntityIndex, anchors: GpuAnchors,
             rows: Any, labels: bool = True) -> tuple[dict[str, _InputValues], dict[str, Any]]:
    """Every value the inputs read, and (with `labels`) every sample target the batch targets read
    (`test.kumo_pytorch.reference_engine.gpu.encoding.targets.sample_targets`; under an inference view of the contract, only the labels a request or
    anchor table supplies, checked as upstream checks them), for the anchors at `rows`; checks read back
    once. Without `labels` (measuring lengths), none."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        _raise_flags,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.targets import (
        sample_targets,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.dated import (
        dated_values,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.events import (
        event_inputs,
    )

    from kumo.pytorch.contract.batch import (
        Composite,
        EntitySource,
        OccurrencesOf,
        Tokens,
    )
    from kumo.pytorch.contract.batch.validation import resolved_fields

    entity, found = index.find(anchors, rows)
    cutoffs = anchors.cutoffs[rows]
    _require_entities(contract, sampler, found, entity, cutoffs, anchors, rows, index)
    flags: list[tuple[Any, Any]] = []
    named, repeated = {}, {}  # (input, selector) -> the entity value it reads ("entity/age" reads "age")
    for name, value in contract.batch.inputs.items():
        if isinstance(value.source, EntitySource):
            for use in value.representation.fields:
                value_name = resolved_fields(contract, name, use.field)[0][0][1]
                (repeated if value_name in contract.sample.dated_entity_values else named)[(name, use.field)] = \
                    value_name
    values = sampler.values(list(dict.fromkeys(named.values())), entity, cutoffs, flags)
    occurrences = dated_values(sampler, list(dict.fromkeys(repeated.values())), entity, cutoffs, flags)
    result: dict[str, _InputValues] = {}
    for (name, selector), value_name in named.items():
        result.setdefault(name, _InputValues(None, {})).values[selector] = values[value_name]
    for (name, selector), value_name in repeated.items():  # a list per anchor: every occurrence, in order
        entry = result.setdefault(name, _InputValues(None, {}))
        entry.values[selector] = occurrences[value_name].values
        entry.dated[selector] = occurrences[value_name]
    for name, value in contract.batch.inputs.items():  # an occurrence field's values (projected when batched)
        if isinstance(value.source, OccurrencesOf):
            result[name] = _InputValues(None, {use.field: occurrences[resolved_fields(contract, name, use.field)[0][0][1]]
                                               .fields[resolved_fields(contract, name, use.field)[0][0][2]]
                                               for use in value.representation.fields})
    for name, value in contract.batch.inputs.items():
        if isinstance(value.source, EntitySource) and isinstance(value.representation.row, (Tokens, Composite)):
            result[name] = _entity_rows(result[name], value.representation, entity.size)
    for name, events in event_inputs(contract, sampler, entity, cutoffs, flags).items():
        result[name] = _InputValues(events.lengths, events.values, events.declared, events.fitting,
                                    components=events.components)
    labels = sample_targets(contract, sampler, anchors, rows, entity, cutoffs, flags) if labels else {}
    _raise_flags(flags)
    return result, labels


def _entity_rows(sampled: _InputValues, representation: Any, count: int) -> _InputValues:
    """An entity token or Composite input's rows: each anchor's entity row, dropped when a field it needs is
    NULL (upstream's ``_eligible``): under ``missing="skip_event"`` any field it encodes, for Composite its
    ``required_fields``. A Composite input's encoders are fitted on every row (`fitting`)."""

    from test.kumo_pytorch.reference_engine.gpu.sampling.dated import (
        Occurrences,
        _taken,
    )

    import cupy

    from kumo.pytorch.contract.batch import Composite

    row = representation.row
    composite = isinstance(row, Composite)
    needed = row.required_fields if composite else \
        tuple(use.field for use in representation.fields) if row.missing == "skip_event" else ()
    needed = tuple(name for name in needed if name not in sampled.dated)  # a list is never NULL
    if not needed:
        return _InputValues(cupy.ones(count, dtype=cupy.int64), sampled.values, dated=sampled.dated)
    keep = cupy.ones(count, dtype=bool)
    for name in needed:
        keep &= sampled.values[name].valid()
    values = {name: Values(value.dtype, value.values[keep], value.valid()[keep], value.texts)
              for name, value in sampled.values.items() if name not in sampled.dated}
    dated = {}
    place = cupy.cumsum(keep) - 1  # each kept anchor's row
    for name, found in sampled.dated.items():  # the occurrences of kept rows, by row
        kept = keep[found.owner]
        owner = place[found.owner[kept]]
        dated[name] = Occurrences(owner, _taken(found.values, kept), found.times[kept],
                                  {key: _taken(field, kept) for key, field in found.fields.items()},
                                  cupy.searchsorted(owner, cupy.arange(int(keep.sum()) + 1, dtype=cupy.int64)).get())
        values[name] = dated[name].values
    return _InputValues(keep.astype(cupy.int64), values, fitting=sampled.values if composite else None, dated=dated)


def _require_entities(contract: DataLoaderSpec, sampler: Sampler, found: Any, entity: Any, cutoffs: Any,
                      anchors: GpuAnchors, rows: Any, index: EntityIndex) -> None:
    """Every anchor's entity as upstream's sample builder requires it: a row of the entity table, passing every
    entity filter (``EntitySpec.filters``, upstream's NULL rules), and in existence before the cutoff
    (``available_from``: its time before the cutoff; a NULL time is refused). One rule code per anchor, read
    back once; the first failing anchor in order is refused, by its first failing rule."""

    from test.kumo_pytorch.reference_engine.gpu.sampling.calculations import (
        _seconds,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.filters import _filter

    import cupy

    spec = contract.sample.entity
    code = cupy.where(found, 0, 1).astype(cupy.int8)  # 1: no row
    if spec.filters or spec.available_from is not None:
        table = sampler.tables[spec.table]
        at = cupy.where(found, entity, 0)
        if spec.filters:
            passing = cupy.ones(at.size, dtype=bool)
            for condition in spec.filters:
                passing &= _filter(table, condition, at)
            code = cupy.where((code == 0) & ~passing, 2, code)  # 2: filtered out
        if spec.available_from is not None:
            column = spec.available_from
            present = table.present(column)
            if present is not None:
                code = cupy.where((code == 0) & ~present[at], 3, code)  # 3: no time
            since = _seconds(table.dtypes[column], table.values(column)[at])
            code = cupy.where((code == 0) & (since >= cutoffs), 4, code)  # 4: not yet in existence
    failing = cupy.flatnonzero(code)
    if not failing.size:
        return
    first = int(failing[0])
    request = anchors.fetch_rows(rows[first:first + 1])[0]
    rule = int(code[first])
    if rule == 1:
        dtype = index.table.dtypes[index.key]
        try:
            dtype.normalize(request.entity_id)  # upstream's check of the ID against the key's type
        except ValueError:
            raise ValueError(f"anchor {request.sample_id!r}: entity ID {request.entity_id!r} is not a valid "
                             f"{dtype.name} value ({index.table.name}.{index.key})") from None
        raise ValueError(f"anchor {request.sample_id!r} names {index.table.name}.{index.key}="
                         f"{request.entity_id!r}, which the table does not hold")
    if rule == 2:
        raise ValueError(f"entity {request.entity_id!r} does not satisfy its configured filters")
    if rule == 3:
        raise ValueError(f"{spec.table!r} row has no timestamp in {spec.available_from!r}")
    raise ValueError("entity did not exist before the requested sample time")
