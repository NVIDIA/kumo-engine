# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How many partitions a split needs: counting each partition's rows until the largest fits."""

from __future__ import annotations

import math
import test.kumo_pytorch.reference_engine.gpu.split.routing as footprint
from collections.abc import Mapping, Sequence
from test.kumo_pytorch.reference_engine.gpu.sampling.paths import Hop
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.contract.dtypes import DType
    from kumo.pytorch.dataset import Parquet

from test.kumo_pytorch.reference_engine.gpu.split import owners as owner_maps
from test.kumo_pytorch.reference_engine.gpu.split import walk
from test.kumo_pytorch.reference_engine.gpu.split.owners import _Owners
from test.kumo_pytorch.reference_engine.gpu.split.partition_plan import (
    PartitionPlan,
)
from test.kumo_pytorch.reference_engine.gpu.split.walk import (
    WALK_FRACTION,
    _human,
    _needed,
    _WalkTooBig,
)

#: How many partition counts a split chosen by `choose_route` tries, when the rows a
#: sample's routes copy in make a partition too big, before it refuses.
SIZING_TRIES = 4


def _fit(plan: PartitionPlan, sources: Mapping[str, tuple[Parquet, Mapping[str, DType]]],
         routes: tuple[tuple[Hop, ...], ...], loads: Mapping[str, footprint._Load], num_partitions: int, room: int | None,
         pair: int | None, chunk_bytes: int,
         budget: int) -> tuple[int, _Owners, list[list[int]], dict[str, list[tuple[str, Any, Any, DType]]],
                               list[dict[str, int]]]:
    """K, its owners, its groups of partitions (`_discover`) and the first group's reached rows.

    With `room`, K grows until each partition fits: it holds at most `room`, and
    two partitions' held bytes and its peak beyond them (`_partition_bytes`) at
    most `pair`. `choose_route` plans K from the average partition, but hashing
    spreads the root rows, and the rows they own, unevenly, and a sample's copied
    rows add more. So each partition's own and copied rows are counted from the
    key columns, and what they take on the GPU estimated from the files' footers
    (`_loads`). With K given (`room` None), nothing is counted. Rows reached by
    many root rows are copied into nearly every partition and do not shrink with
    more partitions: after `SIZING_TRIES` counts, or once K would exceed the root
    rows, the split is refused, before anything is written.
    """

    tried: list[dict[str, int]] = []
    while True:
        owners = _Owners(plan, sources, num_partitions)
        groups, first, own, copied = _discover(plan, sources, owners, routes, chunk_bytes, budget,
                                               count=room is not None)
        if own is None or copied is None or room is None or pair is None:
            return num_partitions, owners, groups, first, tried
        sizes = [footprint._partition_bytes(loads, {name: own[k][name] + copied[k][name] for name in plan.owned})
                 for k in range(num_partitions)]
        need = [max(size["held_bytes"] / room, (2 * size["held_bytes"] + size["extra_bytes"]) / pair)
                for size in sizes]
        largest = max(range(num_partitions), key=need.__getitem__)
        tried.append({"num_partitions": num_partitions, "largest_held_bytes": sizes[largest]["held_bytes"],
                      "largest_extra_bytes": sizes[largest]["extra_bytes"]})
        if need[largest] <= 1:
            return num_partitions, owners, groups, first, tried
        grown = max(num_partitions + 1, math.ceil(num_partitions * need[largest]))
        if len(tried) == SIZING_TRIES or grown > _footer_rows(plan.root, sources):
            held = sizes[largest]["held_bytes"]
            own_bytes = footprint._partition_bytes(loads, own[largest])["held_bytes"]
            detail = ", ".join(f"{name} {_human(footprint._partition_bytes(loads, {name: rows})['held_bytes'])}"
                               for rows, name in sorted(((copied[largest][name], name) for name in plan.owned),
                                                        reverse=True) if rows)
            raise ValueError(
                f"with the sample's routes, partition {largest} of {num_partitions} would hold about "
                f"{_human(held)} with its indexes ({_human(own_bytes)} of its own rows and "
                f"{_human(held - own_bytes)} copied in: {detail}), and need {_human(sizes[largest]['extra_bytes'])} "
                f"more at its peak: more than the {_human(room)} room for one partition, or the {_human(pair)} for "
                f"two and one's peak, after trying {[t['num_partitions'] for t in tried]} partitions. Rows "
                "reached by many root rows (through a shared table, say) are copied into nearly every partition, "
                "so more partitions do not shrink them: leave such routes out of the sample, or pass "
                "num_partitions to split it anyway"
            )
        num_partitions = grown


def _count(plan: PartitionPlan, sources: Mapping[str, tuple[Parquet, Mapping[str, DType]]], owners: _Owners,
           reached: Mapping[str, Sequence[tuple[str, Any, Any, DType]]], chunk_bytes: int,
           copied: list[dict[str, int]], own: list[dict[str, int]] | None = None) -> None:
    """Add each partition's copied rows per split table (and with `own`, its own rows) to the counts given.

    They are counted as `_write_table` would write them, from the key columns
    only. Without `own`, only the tables `reached` copies from are read.
    """

    from test.kumo_pytorch.reference_engine.gpu.data.tables import read_chunks

    for name in plan.owned:
        if own is None and name not in reached:
            continue
        source, columns = sources[name]
        keys = {plan.primary_keys[name]} if name == plan.root else {chain[0].key for chain in plan.owned[name]}
        keys |= {column for column, *_ in reached.get(name, ())}
        for chunk in read_chunks(name, source, {key: columns[key] for key in sorted(keys)}, chunk_bytes=chunk_bytes):
            if chunk.num_rows == 0:
                continue
            _, parts, _, halo_parts = owner_maps._places(name, chunk, owners, reached.get(name, ()))
            for counts, found in ((own, parts), (copied, halo_parts)):
                if counts is not None and found is not None:
                    for k, rows in enumerate(_per_partition(found, owners.k)):
                        counts[k][name] += rows


def _per_partition(parts: Any, num_partitions: int) -> list[int]:
    import pylibcudf as plc

    _, offsets = plc.partitioning.partition(plc.Table([parts]), parts, num_partitions)
    return [offsets[k + 1] - offsets[k] for k in range(num_partitions)]


def _footer_rows(table: str, sources: Mapping[str, tuple[Parquet, Mapping[str, DType]]]) -> int:
    import pyarrow.parquet as pq

    from kumo.pytorch.dataset import _local_parquet_files

    return sum(pq.ParquetFile(path).metadata.num_rows for path in _local_parquet_files(sources[table][0].path))


def _discover(plan: PartitionPlan, sources: Mapping[str, tuple[Parquet, Mapping[str, DType]]], owners: _Owners,
              routes: tuple[tuple[Hop, ...], ...], chunk_bytes: int, budget: int, *,
              count: bool) -> tuple[list[list[int]], dict[str, list[tuple[str, Any, Any, DType]]],
                                    list[dict[str, int]] | None, list[dict[str, int]] | None]:
    """Groups of partitions whose walks fit in `budget` bytes, the first group's reached rows, and counts.

    All partitions are walked together when they fit, which is the usual case
    and costs nothing more. A group that does not fit is halved and walked
    again; a single partition that does not fit is refused. Only the first
    group's reached rows are kept (for the pass that writes every partition's
    own rows); the others are walked again when their copies are written. With
    `count`, each partition's own and copied rows are counted as each group is
    walked (`_count`); otherwise the counts are None.
    """

    everything = list(range(owners.k))
    if not _needed(plan, routes):
        if not count:
            return [everything], {}, None, None
        own = [dict.fromkeys(plan.owned, 0) for _ in everything]
        copied = [dict.fromkeys(plan.owned, 0) for _ in everything]
        _count(plan, sources, owners, {}, chunk_bytes, copied, own)  # nothing copied; own rows hash unevenly
        return [everything], {}, own, copied
    own = [dict.fromkeys(plan.owned, 0) for _ in everything] if count else None
    copied = [dict.fromkeys(plan.owned, 0) for _ in everything] if count else None
    queue, groups = [everything], []
    first: dict[str, list[tuple[str, Any, Any, DType]]] | None = None
    while queue:
        partitions = queue.pop(0)
        try:
            reached = walk._reach(plan, sources, owners, routes, chunk_bytes, partitions=partitions, budget=budget)
        except (_WalkTooBig, MemoryError) as error:
            if len(partitions) == 1:
                raise ValueError(
                    f"the rows the sample's routes reach from partition {partitions[0]}'s {plan.root} rows need "
                    f"more GPU memory than the walk may hold ({_human(budget)}, {WALK_FRACTION:.0%} of the GPU's "
                    f"free memory when the split started): {error}. Split into more partitions (num_partitions), "
                    "or leave out of the sample the routes that reach the most rows"
                ) from error
            half = len(partitions) // 2
            queue[:0] = [partitions[:half], partitions[half:]]
            continue
        groups.append(partitions)
        if copied is not None:
            _count(plan, sources, owners, reached, chunk_bytes, copied, own if len(groups) == 1 else None)
        if first is None:
            first = reached
        del reached
    return groups, first or {}, own, copied
