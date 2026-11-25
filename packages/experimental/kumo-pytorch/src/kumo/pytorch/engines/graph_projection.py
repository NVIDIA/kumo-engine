# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded read-through storage for immutable DiskGraph numeric graph projections."""

from collections import OrderedDict

import numpy as np
import pyarrow as pa


class _Projection:
    """Direct-mapped native node IDs; collisions evict only stored feature values."""

    def __init__(self, columns):
        self.types = {name: col.type for name, col in columns.items()}
        self.values = {}
        self.valid = {}
        row_bytes = 9  # dense ID plus occupied flag
        for ty in self.types.values():
            scalar = ty.value_type if pa.types.is_fixed_size_list(ty) else ty
            width = ty.list_size if pa.types.is_fixed_size_list(ty) else 1
            row_bytes += 1 + width * (scalar.bit_width // 8)
        self.capacity = min(8192, (16 << 20) // row_bytes)
        self.ids = np.zeros(self.capacity, dtype=np.uint64)
        self.occupied = np.zeros(self.capacity, dtype=bool)
        for name, ty in self.types.items():
            vector = pa.types.is_fixed_size_list(ty)
            shape = (self.capacity, ty.list_size) if vector else self.capacity
            scalar = ty.value_type if vector else ty
            self.values[name] = np.empty(shape, dtype=scalar.to_pandas_dtype())
            self.valid[name] = np.empty(self.capacity, dtype=bool)

    def store(self, ids, columns):
        if not self.capacity:
            return
        slots = ids % self.capacity
        keep = slice(None)
        unique_slots, first = np.unique(slots, return_index=True)
        if len(unique_slots) != len(slots):
            slots, keep = unique_slots, first
        for name, column in columns.items():
            self.valid[name][slots] = column.is_valid().to_numpy(zero_copy_only=False)[keep]
            if pa.types.is_fixed_size_list(column.type):
                width = column.type.list_size
                column = column.values.slice(column.offset * width, len(column) * width)
                shape = (len(ids), width)
            else:
                shape = len(ids)
            if column.null_count:
                column = column.fill_null(0)
            self.values[name][slots] = column.to_numpy().reshape(shape)[keep]
        self.ids[slots] = ids[keep]
        self.occupied[slots] = True

    def gather(self, slots):
        result = {}
        for name, ty in self.types.items():
            valid = self.valid[name][slots]
            mask = None if valid.all() else ~valid
            data = self.values[name][slots]
            if pa.types.is_fixed_size_list(ty):
                values = pa.array(data.ravel(), type=ty.value_type)
                result[name] = pa.FixedSizeListArray.from_arrays(
                    values, type=ty, mask=None if mask is None else pa.array(mask)
                )
            else:
                result[name] = pa.array(data, type=ty, mask=mask)
        return result


class GraphProjectionCache:
    """At most four 16 MiB projections per active iterator; all misses use DiskGraph."""

    def __init__(self):
        self.entries = OrderedDict()

    def read(self, key, ids, fetch):
        ids = np.asarray(ids, dtype=np.uint64)
        projection = self.entries.get(key)
        if projection is not None and projection.capacity:
            self.entries.move_to_end(key)
            slots = ids % projection.capacity
            hits = projection.occupied[slots] & (projection.ids[slots] == ids)
            if hits.all():
                return projection.gather(slots)
        else:
            hits = np.zeros(len(ids), dtype=bool)
        missing, inverse = np.unique(ids[~hits], return_inverse=True)
        columns = fetch(missing.tolist())
        if projection is None:
            if len(self.entries) == 4:
                self.entries.popitem(last=False)
            projection = self.entries[key] = _Projection(columns)
        # Gather hits before admitting conflicting keys. Arrow's take restores duplicates,
        # order and nulls. Returned arrays own their data across later cache evictions.
        if hits.any():
            cached = projection.gather(slots[hits])
            order = np.empty(len(ids), dtype=np.intp)
            n = hits.sum()
            order[hits] = np.arange(n)
            order[~hits] = n + inverse
            result = {
                name: pa.concat_arrays([cached[name], column]).take(order)
                for name, column in columns.items()
            }
        else:
            result = {name: column.take(inverse) for name, column in columns.items()}
        projection.store(missing, columns)
        return result
