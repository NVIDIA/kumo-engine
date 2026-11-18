# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Key indexes: a table's rows by key, sorted once on the GPU."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable
if TYPE_CHECKING:

    from test.kumo_pytorch.reference_engine.gpu.data.anchors import GpuAnchors
    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable

__all__ = ['EntityIndex', 'KeyIndex']


class KeyIndex:
    """A table's rows by one key column's identity (`test.kumo_pytorch.reference_engine.gpu.data.keys`; NULL and NaN keys left out, as they match
    nothing), sorted once."""

    def __init__(self, table: GpuTable, column: str) -> None:
        from test.kumo_pytorch.reference_engine.gpu.data.keys import identities

        import cupy

        values, matchable = identities(table, column)
        rows = cupy.arange(table.num_rows, dtype=cupy.int64)
        if matchable is not None:
            rows, values = rows[matchable], values[matchable]
        order = cupy.argsort(values, kind="stable")
        self.keys = values[order]
        self.rows = rows[order]

    def expand(self, owner: Any, keys: Any) -> tuple[Any, Any]:
        """For each (owner, key): every row whose column equals key, as (owner, row) pairs."""

        import cupy

        start = cupy.searchsorted(self.keys, keys, side="left")
        stop = cupy.searchsorted(self.keys, keys, side="right")
        ends = cupy.cumsum(stop - start)
        total = int(ends[-1]) if ends.size else 0
        if total == 0:
            return cupy.zeros(0, dtype=cupy.int64), cupy.zeros(0, dtype=cupy.int64)
        slot = cupy.arange(total, dtype=cupy.int64)
        pair = cupy.searchsorted(ends, slot, side="right")
        within = slot - (ends[pair] - (stop[pair] - start[pair]))
        return owner[pair], self.rows[start[pair] + within]


class EntityIndex:
    """The entity table's rows by primary key identity (`test.kumo_pytorch.reference_engine.gpu.data.keys`: equal exactly when upstream's Python
    values are), sorted once, looked up by binary search.

    Building it checks the keys: a NULL or repeated primary key is refused, since
    an anchor must name exactly one entity row (G-ENTITY-001). A NaN key equals no
    ID (upstream's dict holds NaNs by object), so its row is never found.
    """

    def __init__(self, table: GpuTable, key: str) -> None:
        import weakref
        from test.kumo_pytorch.reference_engine.gpu.data.keys import key_space

        import cupy

        if table.present(key) is not None:
            raise ValueError(f"{table.name}.{key} is the primary key but holds NULLs")
        self.space = key_space(table, key)
        rows, values, matchable = cupy.arange(table.num_rows, dtype=cupy.int64), self.space.at(), self.space.matchable_at()
        if matchable is not None:
            rows, values = rows[matchable], values[matchable]
        order = cupy.argsort(values)
        self.order, self.keys = rows[order], values[order]
        repeated = cupy.flatnonzero(self.keys[1:] == self.keys[:-1])
        if repeated.size:
            from test.kumo_pytorch.reference_engine.gpu.sampling.host_values import (
                _python_at,
            )
            from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
                _column_values,
            )

            row = self.order[int(repeated[0]):int(repeated[0]) + 1]
            value = _python_at(_column_values(table, key, row), 0)
            raise ValueError(f"{table.name}.{key} is the primary key but repeats a value ({value!r})")
        self.table, self.key = table, key
        self._anchors: Any = weakref.WeakKeyDictionary()  # each anchor table's IDs as identities (`find`)

    def find(self, anchors: GpuAnchors, rows: Any) -> tuple[Any, Any]:
        """(row of the entity each anchor at table `rows` (CuPy) names, whether the table holds it), as CuPy arrays.

        IDs match keys as upstream finds an entity: normalized to the key's type, then compared by value
        (`GpuAnchors.entity_identities`, worked out once per anchor table); an ID of another type names none.
        """

        identities = self._anchors.get(anchors)
        if identities is None:
            self._anchors[anchors] = identities = anchors.entity_identities(self.space)
        values, valid = identities
        found_rows, found = self.rows(values[rows])
        return found_rows, found if valid is None else found & valid[rows]

    def rows(self, entity_ids: Any) -> tuple[Any, Any]:
        """(row of each entity key, whether the table holds it), as CuPy arrays."""

        import cupy

        if self.keys.size == 0:
            return cupy.zeros(entity_ids.shape, dtype=cupy.int64), cupy.zeros(entity_ids.shape, dtype=bool)
        position = cupy.minimum(cupy.searchsorted(self.keys, entity_ids), self.keys.size - 1)
        return self.order[position], self.keys[position] == entity_ids
