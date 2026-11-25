# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded read-through storage for immutable DiskGraph numeric graph projections."""

from collections import OrderedDict
from threading import Lock

import numpy as np
import pyarrow as pa

try:
    from diskgraph import _gather_byte_rows
except ImportError:  # Older DiskGraph wheels still support NumPy assembly.
    _gather_byte_rows = None

_CACHE_BYTES = 64 << 20
_MAPPED_BYTES = 16 << 20
_DENSE_BYTES = 32 << 20
_MAX_ROWS = 32768


def contiguous_node_counts(metadata):
    """Find tables whose stored DiskGraph shard ranges cover exactly [0, count)."""
    counts = {}
    for node in metadata.get("node_types", ()):
        if not node.get("has_node_data"):
            continue
        count = 0
        for shard in node.get("shards", ()):
            first, last = shard["min_id"], shard["max_id"]
            if last < first:  # DiskGraph represents an empty shard as [1, 0].
                continue
            if first != count:
                break
            count = last + 1
        else:
            if 0 < count <= _MAX_ROWS and count == node["max_id"] + 1:
                counts[node["name"]] = count
    return counts


def cacheable_projection(columns, names):
    """Cache fixed-width numeric vectors with optional integer columns, including public keys."""
    types = [columns.get(name) for name in names]
    return all(t is not None and (t.shape or t.integer) for t in types) and any(
        t.shape for t in types
    )


class _Projection:
    """Bounded storage for immutable columns, keyed by native node ID."""

    def __init__(self, columns, dense_count, reserve):
        self.types = {name: col.type for name, col in columns.items()}
        row_bytes = 9  # dense ID plus occupied flag
        for ty in self.types.values():
            scalar = ty.value_type if pa.types.is_fixed_size_list(ty) else ty
            width = ty.list_size if pa.types.is_fixed_size_list(ty) else 1
            row_bytes += 1 + width * (scalar.bit_width // 8)
        self.dense = 0 < dense_count <= _MAX_ROWS and dense_count * row_bytes <= _DENSE_BYTES
        capacity = dense_count if self.dense else min(8192, _MAPPED_BYTES // row_bytes)
        self.loaded = 0
        self.evictions = 0
        self.row_bytes = row_bytes
        reserve(capacity * row_bytes)
        self._allocate(capacity)

    def _allocate(self, capacity):
        ids = np.zeros(capacity, dtype=np.uint64)
        occupied = np.zeros(capacity, dtype=bool)
        values, valid = {}, {}
        for name, ty in self.types.items():
            vector = pa.types.is_fixed_size_list(ty)
            shape = (capacity, ty.list_size) if vector else capacity
            scalar = ty.value_type if vector else ty
            values[name] = np.empty(shape, dtype=scalar.to_pandas_dtype())
            valid[name] = np.empty(capacity, dtype=bool)
        self.ids, self.occupied, self.values, self.valid = ids, occupied, values, valid
        self.capacity = capacity
        self.nbytes = capacity * self.row_bytes

    def grow(self, capacity):
        # Rehash only resident IDs. Limit temporary copies of wide vectors during migration.
        old = np.flatnonzero(self.occupied)
        slots, first = np.unique(self.ids[old] % capacity, return_index=True)
        old = old[first]
        ids = self.ids[old]
        values, valid = self.values, self.valid
        self._allocate(capacity)
        chunk = max(1, (256 << 10) // self.row_bytes)
        for start in range(0, len(old), chunk):
            src, dst = old[start : start + chunk], slots[start : start + chunk]
            for name in self.types:
                self.values[name][dst] = values[name][src]
                self.valid[name][dst] = valid[name][src]
        self.ids[slots] = ids
        self.occupied[slots] = True
        self.evictions = None

    @staticmethod
    def unpack(columns):
        arrays = {}
        for name, column in columns.items():
            valid = column.is_valid().to_numpy(zero_copy_only=False)
            shape = (len(column),)
            if pa.types.is_fixed_size_list(column.type):
                width = column.type.list_size
                shape += (width,)
                column = column.values.slice(column.offset * width, len(column) * width)
            if column.null_count:
                column = column.fill_null(0)
            arrays[name] = valid, column.to_numpy().reshape(shape)
        return arrays

    def store(self, ids, arrays):
        if not self.capacity:
            return
        slots = ids if self.dense else ids % self.capacity
        keep = slice(None)
        if self.dense:
            # Callers pass only distinct cache misses.
            self.loaded += len(ids)
        else:
            unique_slots, first = np.unique(slots, return_index=True)
            if len(unique_slots) != len(slots):
                slots, keep = unique_slots, first
            if self.evictions is not None:
                self.evictions += len(ids) - len(slots) + int(self.occupied[slots].sum())
        for name, (valid, data) in arrays.items():
            self.valid[name][slots] = valid[keep]
            self.values[name][slots] = data[keep]
        self.ids[slots] = ids[keep]
        self.occupied[slots] = True

    def gather(self, slots, missing=None, *, as_numpy=False):
        result = {} if as_numpy else GatheredColumns()
        selectors = None
        if slots is not None and missing is not None and _gather_byte_rows is not None:
            positions, inverse, _ = missing
            selectors = slots.astype(np.int64)
            selectors[positions] = ~inverse
        for name, ty in self.types.items():
            bank = self.values[name]
            if slots is None:
                _, inverse, arrays = missing
                new_valid, new_data = arrays[name]
                valid, data = new_valid[inverse], new_data[inverse]
            elif missing is None:
                valid = self.valid[name][slots]
                data = bank[slots]
            else:
                positions, inverse, arrays = missing
                new_valid, new_data = arrays[name]
                valid = self.valid[name][slots]
                valid[positions] = new_valid[inverse]
                if selectors is not None and bank.ndim == 2:
                    raw = _gather_byte_rows(bank.view(np.uint8), new_data.view(np.uint8), selectors)
                    data = np.frombuffer(raw, dtype=bank.dtype).reshape(len(slots), bank.shape[1])
                else:
                    data = bank[slots]
                    data[positions] = new_data[inverse]
            mask = None if valid.all() else ~valid
            if pa.types.is_fixed_size_list(ty):
                # Advanced indexing creates writable buffers owned by this batch alone.
                if mask is not None:
                    data[mask] = 0
            if as_numpy:
                result[name] = data, valid
            elif pa.types.is_fixed_size_list(ty):
                result.vectors[name] = data, valid
                values = pa.array(data.ravel(), type=ty.value_type)
                result[name] = pa.FixedSizeListArray.from_arrays(
                    values, type=ty, mask=None if mask is None else pa.array(mask)
                )
            else:
                result[name] = pa.array(data, type=ty, mask=mask)
        return result


class GatheredColumns(dict):
    """Arrow feature columns with owned buffers for direct raw-vector tensor output."""

    def __init__(self):
        super().__init__()
        self.vectors = {}


class GraphProjectionCache:
    """At most 64 MiB across four projections for one immutable DiskGraph anchor task."""

    def __init__(self):
        self.entries = OrderedDict()
        self._lock = Lock()

    def _reserve(self, nbytes):
        while (
            len(self.entries) >= 4
            or sum(p.nbytes for p in self.entries.values()) + nbytes > _CACHE_BYTES
        ):
            self.entries.popitem(last=False)

    def read(self, key, ids, fetch, *, dense_count=0, as_numpy=False):
        # Multiple iterators can use a materialized task. Slot IDs, values and validity
        # must stay consistent throughout a gather, including while native IO releases the GIL.
        with self._lock:
            return self._read(key, ids, fetch, dense_count, as_numpy)

    def _read(self, key, ids, fetch, dense_count, as_numpy):
        ids = np.asarray(ids, dtype=np.uint64)
        projection = self.entries.get(key)
        if projection is not None and projection.capacity:
            self.entries.move_to_end(key)
            if projection.dense and ids.max(initial=0) >= projection.capacity:
                # A bound sampler view may contain additional IDs outside the base index.
                projection.dense = False
            if projection.dense and projection.loaded == projection.capacity:
                return projection.gather(ids, as_numpy=as_numpy)
            if projection.evictions is not None and projection.evictions * 4 >= projection.capacity:
                # Grow only after repeated conflicts and only into currently spare space.
                # Both the old and new arrays fit the task budget while rehashing.
                free = _CACHE_BYTES - sum(p.nbytes for p in self.entries.values())
                capacity = min(_MAX_ROWS, projection.capacity * 3, free // projection.row_bytes)
                if capacity >= projection.capacity * 2:
                    projection.grow(capacity)
                else:
                    projection.evictions = 0
            slots = ids if projection.dense else ids % projection.capacity
            hits = projection.occupied[slots] & (projection.ids[slots] == ids)
            if hits.all():
                return projection.gather(slots, as_numpy=as_numpy)
        else:
            hits = np.zeros(len(ids), dtype=bool)
        missing, inverse = np.unique(ids[~hits], return_inverse=True)
        columns = fetch(missing.tolist())
        if projection is None:
            if len(ids) and ids.max() >= dense_count:
                dense_count = 0
            projection = self.entries[key] = _Projection(columns, dense_count, self._reserve)
        # Gather hits before admitting conflicting keys. Patch missing rows directly in
        # the output buffer, avoiding a second full copy of wide vectors via Arrow take.
        arrays = projection.unpack(columns)
        if projection.dense:
            projection.store(missing, arrays)
            return projection.gather(ids, as_numpy=as_numpy)
        result = projection.gather(
            slots if hits.any() else None, (~hits, inverse, arrays), as_numpy=as_numpy
        )
        projection.store(missing, arrays)
        return result
