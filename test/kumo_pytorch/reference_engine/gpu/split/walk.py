# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The halo walk: which rows each partition's samples reach, within a GPU memory budget."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from test.kumo_pytorch.reference_engine.gpu.sampling.paths import Hop
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.contract.dtypes import DType
    from kumo.pytorch.dataset import Parquet

from test.kumo_pytorch.reference_engine.gpu.split.owners import (
    _Formed,
    _formed,
    _gather,
    _joinable,
    _Owners,
    _union,
)
from test.kumo_pytorch.reference_engine.gpu.split.partition_plan import (
    PartitionPlan,
    _downward,
)

#: The share of the GPU's free memory, when a split starts, that walking a sample's
#: routes may hold at once. Past it, the copies are worked out for groups of
#: partitions, one group at a time.
WALK_FRACTION = 0.25


class _WalkTooBig(Exception):
    """Walking the routes for a group of partitions would hold more GPU memory than its budget."""


def _walk_budget() -> int:
    """`WALK_FRACTION` of the GPU's free memory now, an RMM pool's memory included."""

    from test.kumo_pytorch.reference_engine.gpu.data.memory import (
        ensure_pool,
        free_bytes,
    )

    ensure_pool()
    return int(free_bytes() * WALK_FRACTION)


def _copies(plan: PartitionPlan, prefix: tuple[Hop, ...]) -> bool:
    """Whether the rows `prefix` reaches may belong to another partition: a split table, not down a chain."""

    return prefix[-1].table in plan.owned and not _downward(plan.owned, prefix)


def _needed(plan: PartitionPlan, routes: tuple[tuple[Hop, ...], ...]) -> set[tuple[Hop, ...]]:
    """Every beginning of a route that must be walked: up to its last hop whose rows may be copied."""

    needed = set()
    for route in routes:
        last = max((i for i in range(len(route)) if _copies(plan, route[:i + 1])), default=-1)
        needed.update(route[:n] for n in range(1, last + 2))
    return needed


def _reach(plan: PartitionPlan, sources: Mapping[str, tuple[Parquet, Mapping[str, DType]]], owners: _Owners,
           routes: tuple[tuple[Hop, ...], ...], chunk_bytes: int, *, partitions: Sequence[int] | None = None,
           budget: int | None = None) -> dict[str, list[tuple[str, Any, Any, DType]]]:
    """For each split table, (column, values, partition, the values' type): partition k needs the rows whose column
    is a value paired with k (values of another type than the column's meet it as Python's equal values do).

    Every route is walked from the root rows of `partitions` (all when
    omitted) at once, each value carrying the partition of the root row it
    came from. Routes sharing a beginning share its walk. A route's end in
    shared tables is not walked: those rows are on every GPU, and nothing
    after them is copied. Nor are the rows of a route that only walks down
    chains from the root (`_downward`): they are the partition's own. A step's
    keys are freed once the steps after it no longer need them; with `budget`,
    `_WalkTooBig` is raised before the walk holds more GPU bytes than that.
    """

    needed = _needed(plan, routes)
    if not needed:
        return {}
    children: dict[tuple[Hop, ...], list[tuple[Hop, ...]]] = defaultdict(list)
    for prefix in sorted(needed, key=len):
        children[prefix[:-1]].append(prefix)
    keys, parts = owners.key_map(plan.root, ())
    if partitions is not None and len(partitions) < owners.k:
        keys, parts = _only(keys, parts, partitions)
    walked: dict[tuple[Hop, ...], tuple[str, str, tuple[Any, Any, DType] | None]] = {
        (): (plan.root, plan.primary_keys[plan.root], (keys, parts, sources[plan.root][1][plan.primary_keys[plan.root]]))}
    pending: list[tuple[Hop, ...]] = [()]
    while pending:
        parent = pending.pop()
        table, column, pairs = walked[parent]
        assert pairs is not None
        if children[parent]:
            held = _bytes(*(c for _, _, kept in walked.values() if kept is not None for c in kept[:2]))
            ahead = _project(table, column, pairs, {child[-1].from_column for child in children[parent]}, sources,
                             chunk_bytes, held, budget)
            for child in children[parent]:
                hop = child[-1]
                walked[child] = (hop.table, hop.column, ahead[hop.from_column])
                pending.append(child)
        if not parent or not _copies(plan, parent):
            walked[parent] = (table, column, None)  # its children have their keys; nothing copies its rows
    # values reaching one column, by their own type (routes of other types reach it apart), merged per type; those
    # of a type never equal to the column's dropped (they meet nothing), those of another number type turned into
    # the column's exact form once (`_Formed`), as every chunk's join reads them
    from test.kumo_pytorch.reference_engine.gpu.data.keys import never_equal

    selectors: dict[str, dict[tuple[str, DType], list[tuple[Any, Any]]]] = defaultdict(lambda: defaultdict(list))
    for table, column, kept in walked.values():
        if kept is not None:
            selectors[table][(column, kept[2])].append(kept[:2])
    reached: dict[str, list[tuple[str, Any, Any, DType]]] = {}
    for table, by in selectors.items():
        for (column, dtype), found in by.items():
            own = sources[table][1][column]
            if never_equal(own, dtype):
                continue
            values, parts = _union([v for v, _ in found], [p for _, p in found])
            if _joinable(own, dtype):
                reached.setdefault(table, []).append((column, values, parts, dtype))
            else:
                reached.setdefault(table, []).append((column, _formed(values, parts, dtype, own), None, own))
    if budget is not None:
        held = _bytes(*(c for selected in reached.values() for _, keys, parts, _ in selected
                        for c in (keys.columns() if isinstance(keys, _Formed) else (keys, parts))))
        if held > budget:
            raise _WalkTooBig(f"{_human(held)} held by the rows the routes reach")
    return reached


def _only(keys: Any, parts: Any, partitions: Sequence[int]) -> tuple[Any, Any]:
    """The (key, partition) pairs whose partition is one of `partitions`."""

    import pyarrow as pa
    import pylibcudf as plc

    wanted = plc.Column.from_arrow(pa.array(sorted(partitions), pa.int32()))
    kept = plc.stream_compaction.apply_retention_mask(plc.Table([keys, parts]), plc.search.contains(wanted, parts))
    return kept.columns()[0], kept.columns()[1]


def _bytes(*columns: Any) -> int:
    """GPU bytes held by `columns`, each counted once."""

    return sum(column.device_buffer_size() for column in {id(column): column for column in columns}.values())


def _project(table: str, column: str, pairs: tuple[Any, Any, DType], wanted: set[str],
             sources: Mapping[str, tuple[Parquet, Mapping[str, DType]]], chunk_bytes: int, held: int = 0,
             budget: int | None = None) -> dict[str, tuple[Any, Any, DType]]:
    """For each column in `wanted`: (its values, partition, their type) in the rows of `table` whose `column` is
    paired with that partition (`pairs`: values, partitions, and the values' type).

    With `budget`, the pieces found so far are merged (duplicates dropped)
    whenever they and the `held` bytes would pass it, and `_WalkTooBig` is
    raised if they still do.
    """

    from test.kumo_pytorch.reference_engine.gpu.data.keys import never_equal
    from test.kumo_pytorch.reference_engine.gpu.data.tables import read_chunks

    values, parts, dtype = pairs
    source, columns = sources[table]
    # the column itself too, when wanted: the values of the rows met, not the values carried (one may meet no row)
    others = sorted(wanted)
    if not _joinable(columns[column], dtype) and not never_equal(columns[column], dtype):
        values, parts = _formed(values, parts, dtype, columns[column]), None  # once, not per chunk
        held += _bytes(*values.columns())
    found: dict[str, tuple[list[Any], list[Any]]] = {name: ([], []) for name in others}

    def size() -> int:
        return _bytes(*(piece for pieces in found.values() for side in pieces for piece in side))

    for chunk in read_chunks(table, source, {name: columns[name] for name in sorted({column, *others})},
                             chunk_bytes=chunk_bytes):
        rows, chunk_parts = _Owners._join(chunk.columns[column], values, parts, columns[column], dtype)
        for name in others:
            found[name][0].append(_gather(chunk.columns[name], rows))
            found[name][1].append(chunk_parts)
        del rows, chunk_parts, chunk
        if budget is not None and held + size() > budget:
            for name in others:
                merged = _union(*found[name])
                found[name] = ([merged[0]], [merged[1]])
            if held + size() > budget:
                raise _WalkTooBig(f"{_human(held + size())} held at {table}.{', '.join(others)}")
    return {name: (*_union(*found[name]), columns[name]) for name in others}


def _human(size: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if size >= scale:
            return f"{size / scale:.2f} {unit}"
    return f"{size:.0f} B"
