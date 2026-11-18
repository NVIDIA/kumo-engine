# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token inputs, final ID spaces and flattened histories, on the GPU.

Upstream Kumo PyTorch's rules (``assembly.py`` ``_allocate_spaces``, ``_token_rows``,
``_token_output``; ``output.py`` ``flatten_rows`` and ``TokenRows.output``;
G-HISTORY-001):

- **ID spaces.** Each ``IdSpace`` numbers its declared special roles first (in
  upstream's role order, then the others by name), then, member by member in
  the order of their identities, every local ID the member can give, in
  order: a local ID whose role the space declares (a vocabulary's missing or
  unknown ID) takes that special's number, the others the next free ones. A
  member is an encoder (an input's or a target's), or a ``Composite`` input's
  outer IDs (missing 0, unknown 1, then its tuples' IDs). A field (or Composite input) that names a
  space has its IDs translated into it.
- **Token rows.** A row (an event, or the entity's row) becomes its fields' IDs
  in the declared field order; a Text field gives each of its pieces, in order
  (``position_ids``: the piece's place in its field; 0 for other fields). A
  field its row's component does not declare
  gives no token. A NULL gives no token under ``missing="omit"`` (or
  ``"skip_event"``, which has dropped such rows already), its missing ID under
  ``"emit"``, and is refused under ``"error"``. ``max_tokens`` then keeps that
  many tokens of the row, from the end ``keep`` names.
- **Flatten.** A sample's events become one sequence: each event's tokens, a
  separator after each event (``"after_each"``) or between events
  (``"between"``), then the ``before`` and ``after`` markers around it all.
  The markers come out of the ``max_tokens`` budget first; past the budget, the
  end ``keep`` names is kept, cut at an event boundary when one leaves something
  (``prefer_event_boundary``), else inside the event. Markers have no event
  (``source_event_indices`` -1).

Everything here works on all of a call's rows at once; nothing is read back per
row or per sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
    encoder_identity,
    graph_identity,
)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
        CompositeVocabulary,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        GpuEncoder,
    )

    from kumo.pytorch.encoding import FittedSpace
    from kumo.pytorch.spec import DataLoaderSpec

__all__ = ["Flat", "TokenField", "TokenRows", "allocate_spaces", "flatten", "per_sample", "token_rows", "translated"]


def allocate_spaces(contract: DataLoaderSpec, encoders: dict[str, GpuEncoder],
                    composites: dict[str, CompositeVocabulary],
                    field_maps: dict[str, dict[tuple[str, ...], int]] | None = None) -> dict[str, FittedSpace]:
    """Every ``IdSpace``'s final numbering, as upstream's ``FittedSpace`` (its ``_allocate_spaces``); a
    ``field_space`` holds its field-name dictionary (`field_maps`, `test.kumo_pytorch.reference_engine.gpu.encoding.field_names`) as one member, 0 its
    pad."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
        composite_identity,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.field_names import (
        fields_identity,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.targets import (
        target_identity,
        target_uses,
    )

    from kumo.pytorch.contract.batch import Composite
    from kumo.pytorch.contract.batch.tokens import ROLE_ORDER
    from kumo.pytorch.encoding import FittedSpace

    batch = contract.batch
    members: dict[str, dict[str, tuple[tuple[int, ...], dict[str, int]]]] = {name: {} for name in batch.id_spaces}
    for input_name, value in batch.inputs.items():
        for use in getattr(value.representation, "fields", ()):  # graph inputs: none
            if use.id_space:
                encoder = encoders[encoder_identity(input_name, use)]
                members[use.id_space][encoder.identity] = (encoder.local_ids, encoder.roles)
        field_space = getattr(value.representation, "field_space", None)
        if field_space:
            dictionary = value.representation.field_names
            members[field_space][fields_identity(dictionary)] = ((0, *(field_maps or {})[dictionary].values()),
                                                                 {"pad": 0})
        row = getattr(value.representation, "row", None)
        if isinstance(row, Composite) and row.id_space:
            members[row.id_space][composite_identity(input_name)] = (
                (0, 1, *composites[input_name].all_ids), {"missing": 0, "unknown": 1})
        for table, group in getattr(value.representation, "node_fields", {}).items():  # graph inputs
            for use in group.fields:
                if use.id_space:
                    encoder = encoders[graph_identity(input_name, table, use)]
                    members[use.id_space][encoder.identity] = (encoder.local_ids, encoder.roles)
    for name, _, use in target_uses(contract):
        if use.id_space:
            encoder = encoders[target_identity(name, use)]
            members[use.id_space][encoder.identity] = (encoder.local_ids, encoder.roles)
    spaces = {}
    for name, definition in batch.id_spaces.items():
        if definition.special_ids:
            roles = dict(definition.special_ids)
            cursor = max(roles.values()) + 1
        else:
            role_names = [role for role in ROLE_ORDER if role in definition.specials] + sorted(
                set(definition.specials) - set(ROLE_ORDER))
            roles = {role: index for index, role in enumerate(role_names)}
            cursor = len(roles)
        mappings = {}
        for identity, (ids, local_roles) in sorted(members[name].items()):
            inverse = {value: role for role, value in local_roles.items()}
            mapping = {}
            for index in sorted(set(ids)):
                role = inverse.get(index)
                if role in roles:
                    mapping[index] = roles[role]
                else:
                    mapping[index] = cursor
                    cursor += 1
            mappings[identity] = mapping
        spaces[name] = FittedSpace(roles=roles, members=mappings, size=cursor)
    return spaces


def translated(space: FittedSpace, identity: str, ids: Any) -> Any:
    """An encoder's local IDs (CuPy int64) in `space`'s final numbering. Every local ID an encoder gives has
    a number there (`allocate_spaces`), so a sorted lookup finds each."""

    import cupy

    mapping = space.members[identity]
    if not mapping or ids.size == 0:
        return ids
    local = sorted(mapping)
    keys = cupy.asarray(local, dtype=cupy.int64)
    final = cupy.asarray([mapping[index] for index in local], dtype=cupy.int64)
    return final[cupy.minimum(cupy.searchsorted(keys, ids), keys.size - 1)]


@dataclass
class TokenRows:
    """A token input's rows as tokens, for a call's rows.

    :param values: each token's final ID (CuPy int64), row by row, fields in their declared order.
    :param bounds: each row's tokens (CuPy int64, rows + 1).
    :param rows: each token's row (CuPy int64).
    :param slots: each token's field, by its place in the declared order (CuPy int64; upstream's ``slots``).
    :param positions: each token's place among its field's pieces (CuPy int64; 0 but for Text).
    :param occurrences: each token's occurrence among its dated field's (CuPy int64; 0 for other fields).
    :param fields: each token's field-name ID (CuPy int64; `test.kumo_pytorch.reference_engine.gpu.encoding.field_names`), or None without one.
    """

    values: Any
    bounds: Any
    rows: Any
    slots: Any = None
    positions: Any = None
    occurrences: Any = None
    fields: Any = None


@dataclass(frozen=True)
class TokenField:
    """One field of a token input, encoded for a call's rows.

    :param name: the field's selector, as the input names it.
    :param ids: final IDs (CuPy int64): a NULL's is its encoder's missing ID, translated.
    :param valid: whether each row's value is not NULL (CuPy bool).
    :param declared: whether each row's component declares the field (CuPy bool), or None for every row.
    :param pieces: a Text field's ``piece_offsets`` (CuPy int64, items + 1), its `ids` then every item's pieces;
        None for a field of one ID per item.
    :param items: a dated field's occurrences of each row (CuPy int64 offsets, rows + 1), `valid` (and `ids`
        unless `pieces`) then per occurrence; None for one item (the value) per row.
    :param times: a dated field's occurrence times (CuPy int64, per item), for ``occurrence_order="chronological"``.
    """

    name: str
    ids: Any
    valid: Any
    declared: Any
    pieces: Any = None
    items: Any = None
    times: Any = None


def token_rows(input_name: str, row: Any, fields: list[TokenField], count: int, owner: Any, cuts: Any,
               flags: list[tuple[Any, Any]]) -> TokenRows:
    """`count` rows as tokens under upstream's ``Tokens`` rules (`row`: the input's ``Tokens``).

    `owner` (CuPy int64) is each row's anchor position in the call and `cuts` (host int64) where each batch
    starts, then the end: a NULL under ``missing="error"`` is reported for the field and batch upstream would
    report first (the earliest batch, then the first field in the declared order).
    """

    import cupy

    parts, bad = [], []
    for rank, field in enumerate(fields):
        if field.items is None:  # one item per row: its value
            item_row = cupy.arange(count, dtype=cupy.int64)
            occurrence = cupy.zeros(count, dtype=cupy.int64)
        else:
            item_row = cupy.searchsorted(field.items[1:], cupy.arange(int(field.valid.size), dtype=cupy.int64),
                                         side="right")
            occurrence = cupy.arange(int(field.valid.size), dtype=cupy.int64) - field.items[item_row]
        present = (cupy.ones(count, dtype=bool) if field.declared is None else field.declared)[item_row]
        if row.missing == "error":
            null_rows = cupy.zeros(count, dtype=bool)
            null_rows[item_row[present & ~field.valid]] = True
            bad.append(null_rows)
        include = present & field.valid if row.missing in {"omit", "skip_event"} else present
        lengths = cupy.where(include, 1 if field.pieces is None else cupy.diff(field.pieces), 0)
        ends = cupy.cumsum(lengths)
        total = int(ends[-1]) if ends.size else 0
        slot = cupy.arange(total, dtype=cupy.int64)
        item = cupy.searchsorted(ends, slot, side="right")  # item by item, piece by piece
        piece = slot - (ends[item] - lengths[item]) if total else slot
        values = field.ids[item] if field.pieces is None else field.ids[field.pieces[item] + piece]
        when = field.times[item] if field.times is not None else cupy.zeros(total, dtype=cupy.int64)
        parts.append((item_row[item], cupy.full(total, rank, dtype=cupy.int64), slot, piece, values,
                      occurrence[item], when, cupy.full(total, field.times is not None, dtype=bool)))
    if bad:
        stacked = cupy.stack(bad)
        flags.append((lambda: _first_null(input_name, fields, stacked, owner, cuts), stacked.any()))
    rows, slots, sequence, positions, values, occurrences, times, dated = (
        cupy.concatenate([part[index] for part in parts]) for index in range(8))
    if row.occurrence_order == "chronological":  # undated fields first, then occurrences by time
        order = cupy.lexsort(cupy.stack([sequence, slots, times, dated.astype(cupy.int64), rows]))
    else:  # field by field, each its occurrences and pieces in order
        order = cupy.lexsort(cupy.stack([sequence, slots, rows]))
    rows, slots, positions, values, occurrences = (rows[order], slots[order], positions[order], values[order],
                                                   occurrences[order])
    per_row = cupy.bincount(rows, minlength=count)[:count] if rows.size else cupy.zeros(count, dtype=cupy.int64)
    if row.max_tokens is not None:  # the row's first or last `max_tokens`, every piece counted
        starts = cupy.zeros(count + 1, dtype=cupy.int64)
        cupy.cumsum(per_row, out=starts[1:])
        within = cupy.arange(rows.size, dtype=cupy.int64) - starts[rows]
        keep = within < row.max_tokens if row.keep == "first" else within >= per_row[rows] - row.max_tokens
        rows, slots, positions, values, occurrences = (rows[keep], slots[keep], positions[keep], values[keep],
                                                       occurrences[keep])
        per_row = cupy.minimum(per_row, row.max_tokens)
    bounds = cupy.zeros(count + 1, dtype=cupy.int64)
    cupy.cumsum(per_row, out=bounds[1:])
    return TokenRows(values, bounds, rows, slots, positions, occurrences)


def _first_null(input_name: str, fields: list[TokenField], bad: Any, owner: Any, cuts: Any) -> str:
    """Upstream's message for the NULL it would meet first: in the earliest batch, then the earliest field."""

    import cupy
    import numpy as np

    big = cupy.iinfo(cupy.int64).max
    first = cupy.where(bad, owner[None, :], big).min(axis=1).get().tolist()
    _, rank = min((int(np.searchsorted(cuts, value, side="right")), rank) for rank, value in enumerate(first)
                  if value != big)
    return f"input {input_name!r}, field {fields[rank].name!r}: NULL under missing='error'"


@dataclass
class Flat:
    """Tokens as one sequence per sample.

    :param values: each token's final ID (CuPy int64), sample by sample.
    :param bounds: each sample's tokens (host int64, samples + 1).
    :param rows: each token's row (CuPy int64), or -1 for a marker.
    :param sources: each token's place among the `TokenRows` it came from (CuPy int64), or -1 for a marker.
    :param positions: each token's place among its field's pieces (CuPy int64; 0 for a marker).
    """

    values: Any
    bounds: Any
    rows: Any
    sources: Any = None
    positions: Any = None


def per_sample(tokens: TokenRows, row_bounds: Any) -> Flat:
    """Each sample's rows' tokens, one after another (an entity token input: one row per sample at most)."""

    import cupy

    bounds = tokens.bounds[cupy.asarray(row_bounds)]
    return Flat(tokens.values, bounds.get(), tokens.rows, cupy.arange(tokens.values.size, dtype=cupy.int64),
                tokens.positions)


def flatten(tokens: TokenRows, row_bounds: Any, config: Any, roles: Any) -> Flat:
    """Each sample's events as one sequence, under upstream's ``Flatten`` rules (`row_bounds`: each sample's
    events, host int64; `roles`: the output space's special numbers)."""

    import cupy

    count = row_bounds.size - 1
    events = int(row_bounds[-1])
    sample_rows = cupy.asarray(row_bounds)
    lengths = cupy.diff(tokens.bounds)
    owner = cupy.searchsorted(sample_rows[1:], cupy.arange(events, dtype=cupy.int64), side="right")
    if config.separator is None:
        separated = cupy.zeros(events, dtype=cupy.int64)
    elif config.separator_placement == "after_each":
        separated = cupy.ones(events, dtype=cupy.int64)
    else:  # between: after every event but its sample's last
        separated = (cupy.arange(events, dtype=cupy.int64) + 1 < sample_rows[owner + 1]).astype(cupy.int64)
    starts = cupy.zeros(events + 1, dtype=cupy.int64)  # each event's segment in the samples' streams
    cupy.cumsum(lengths + separated, out=starts[1:])
    base = starts[sample_rows[:-1]]
    size = starts[sample_rows[1:]] - base
    before, after = int(config.before is not None), int(config.after is not None)
    budget = config.max_tokens - before - after
    over = size > budget
    low = cupy.zeros(count, dtype=cupy.int64)
    high = size.copy()
    if config.keep == "last":
        low = cupy.where(over, size - budget, 0)
        if config.prefer_event_boundary and events:
            # the first event boundary at or after the cut, if it leaves something
            found = cupy.searchsorted(starts[:-1], base + low, side="left")
            boundary = cupy.where(found < events, starts[cupy.minimum(found, events - 1)] - base, size)
            low = cupy.where(over & (boundary < size), boundary, low)
    else:
        high = cupy.where(over, budget, size)
        if config.prefer_event_boundary and events:
            # the last event boundary at or before the cut, unless it is the start (keep part of the event)
            found = cupy.searchsorted(starts[:-1], base + budget, side="right") - 1
            boundary = starts[cupy.maximum(found, 0)] - base
            high = cupy.where(over & (boundary != 0), boundary, high)
    kept = before + (high - low) + after
    bounds = cupy.zeros(count + 1, dtype=cupy.int64)
    cupy.cumsum(kept, out=bounds[1:])
    host = bounds.get()
    total = int(host[-1])
    slot = cupy.arange(total, dtype=cupy.int64)
    sample = cupy.searchsorted(bounds[1:], slot, side="right")
    offset = slot - bounds[sample]
    is_before = (offset == 0) if before else cupy.zeros(total, dtype=bool)
    is_after = (offset == kept[sample] - 1) if after else cupy.zeros(total, dtype=bool)
    content = ~is_before & ~is_after
    stream = base[sample] + low[sample] + offset - before
    event = cupy.clip(cupy.searchsorted(starts, stream, side="right") - 1, 0, max(events - 1, 0))
    within = stream - starts[event] if events else cupy.zeros(total, dtype=cupy.int64)
    is_token = content & (within < lengths[event]) if events else cupy.zeros(total, dtype=bool)
    if tokens.values.size:
        position = cupy.clip(tokens.bounds[event] + within, 0, tokens.values.size - 1)
        values = cupy.where(is_token, tokens.values[position], 0)
    else:
        position = cupy.zeros(total, dtype=cupy.int64)
        values = cupy.zeros(total, dtype=cupy.int64)
    for marker, role in ((is_before, config.before), (content & ~is_token, config.separator),
                         (is_after, config.after)):
        if role is not None:
            values = cupy.where(marker, roles[role], values)
    pieces = cupy.where(is_token, tokens.positions[position], 0) if tokens.values.size else \
        cupy.zeros(total, dtype=cupy.int64)
    return Flat(values, host, cupy.where(is_token, event, -1), cupy.where(is_token, position, -1), pieces)
