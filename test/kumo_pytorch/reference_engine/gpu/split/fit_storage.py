# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact host fitting storage used by budgeted split materialization.

RAM is used until an explicit retained-entry budget is reached, then a private
SQLite B-tree merges counts on disk. Keys are canonical bytes supplied by the
caller; sortable uint64 keys support exact numeric order statistics too.

The budget covers conservatively charged retained entries, not process RSS,
caller input/output, or SQLite workspace (a separate cache allowance). No
automatic sizing, background writes, checkpoints or GPU finalization yet.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

_MAX_COUNT = (1 << 63) - 1


class FitCounts:
    """Exact byte-key counts. Use as a context manager; None disables spilling.

    Mutating during iteration is unsupported. Failed additions poison the store;
    close and restart rather than retrying potentially partially consumed input.
    Output is byte-sorted, not frequency-sorted. Encoding/tie policy belongs to
    the caller, so different types must have distinct canonical encodings.
    """

    def __init__(self, memory_limit: int, *, spill_directory: str | Path | None = None,
                 cache_bytes: int = 64 * 1024) -> None:
        if type(memory_limit) is not int or memory_limit <= 0:
            raise ValueError('memory_limit must be a positive byte count')
        if type(cache_bytes) is not int or cache_bytes < 1024:
            raise ValueError('cache_bytes must be at least 1024')
        self.memory_limit, self.cache_bytes = memory_limit, cache_bytes
        self.spill_directory = spill_directory
        self.retained_bytes = self.peak_retained_bytes = self.total = 0
        self._counts: dict[bytes, int] = {}
        self._db: sqlite3.Connection | None = None
        self._temporary: tempfile.TemporaryDirectory | None = None
        self._closed = self._failed = False

    def __enter__(self) -> FitCounts:
        self._check()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _check(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError('fitting accumulator is closed or failed')

    @property
    def spilled(self) -> bool:
        return self._db is not None

    @property
    def disk_bytes(self) -> int:
        if self._temporary is None:
            return 0
        return sum(p.stat().st_size for p in Path(self._temporary.name).iterdir() if p.is_file())

    @staticmethod
    def _charge(key: bytes) -> int:
        # Key, int64 count, dict growth and sorting-pointer headroom.
        return sys.getsizeof(key) + 192

    def _spill(self) -> None:
        if self.spill_directory is None:
            raise MemoryError('fitting state exceeds memory_limit; set spill_directory to permit disk use')
        temporary = tempfile.TemporaryDirectory(prefix='kumo-fit-', dir=self.spill_directory)
        db = None
        try:
            db = sqlite3.connect(str(Path(temporary.name) / 'counts.sqlite'))
            db.execute(f'PRAGMA cache_size=-{max(1, self.cache_bytes // 1024)}')
            db.execute('PRAGMA mmap_size=0')
            db.execute('PRAGMA temp_store=FILE')
            db.execute('CREATE TABLE counts (key BLOB PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID')
            db.executemany('INSERT INTO counts VALUES (?, ?)', self._counts.items())
            db.commit()
        except BaseException:
            if db is not None:
                db.close()
            temporary.cleanup()
            raise
        self._temporary, self._db = temporary, db
        self._counts.clear()
        self.retained_bytes = 0

    def add(self, records: Iterable[tuple[bytes, int]]) -> None:
        """Consume records without buffering input. Refuse int64 count overflow."""
        self._check()
        try:
            for key, count in records:
                if type(key) is not bytes or type(count) is not int or count <= 0:
                    raise ValueError('records require bytes keys and positive integer counts')
                if count > _MAX_COUNT - self.total:
                    raise OverflowError('fitting observation count exceeds int64')
                charge = self._charge(key)
                if charge > self.memory_limit:
                    raise MemoryError('one fitting key exceeds memory_limit')
                if self._db is None and key not in self._counts:
                    if hasattr(self, '_budget'):
                        self._budget.reserve(self, charge)
                    if self._db is not None:
                        pass
                    elif self.retained_bytes + charge > self.memory_limit:
                        self._spill()
                    else:
                        self.retained_bytes += charge
                        self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
                if self._db is None:
                    self._counts[key] = self._counts.get(key, 0) + count
                else:
                    self._db.execute('INSERT INTO counts VALUES (?, ?) ON CONFLICT(key) DO UPDATE '
                                     'SET count=count+excluded.count', (key, count))
                self.total += count
            if self._db is not None:
                self._db.commit()
        except BaseException:
            self._failed = True
            raise

    def items(self) -> Iterator[tuple[bytes, int]]:
        """Stream merged counts. Output storage belongs to the caller."""
        self._check()
        if self._db is None:
            yield from sorted(self._counts.items())
        else:
            cursor = self._db.execute('SELECT key, count FROM counts ORDER BY key')
            try:
                yield from cursor
            finally:
                cursor.close()

    def order_statistics(self, positions: Iterable[int]) -> list[bytes]:
        """Exact zero-based weighted ranks, returned in requested order.

        Only requested ranks/results are materialized. Quantile interpolation
        remains the caller's existing numerical policy.
        """
        self._check()
        requested = list(positions)
        if any(type(p) is not int or not 0 <= p < self.total for p in requested):
            raise ValueError('order-statistic position is outside the observations')
        result = [b''] * len(requested)
        pending = iter(sorted(enumerate(requested), key=lambda pair: pair[1]))
        target = next(pending, None)
        if target is None:
            return result
        seen = 0
        rows = self.items()
        try:
            for key, count in rows:
                seen += count
                while target is not None and target[1] < seen:
                    result[target[0]] = key
                    target = next(pending, None)
                if target is None:
                    break
        finally:
            rows.close()
        return result

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._db is not None:
                self._db.close()
        finally:
            self._db = None
            self._counts.clear()
            self.retained_bytes = 0
            self._closed = True
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None


class FitBudget:
    """One retained-entry allowance shared by every accumulator in a fit.

    Evict the largest resident accumulator when necessary. SQLite caches (64 KiB
    per spilled accumulator), incoming chunks and final output are separate.
    """

    def __init__(self, limit: int, directory: str | Path | None = None):
        if type(limit) is not int or limit <= 0:
            raise ValueError('fit_memory_limit must be a positive integer byte count')
        self.limit, self.directory = limit, directory
        self.stores: list[FitCounts] = []
        self.peak_retained_bytes = 0

    def counts(self) -> FitCounts:
        store = FitCounts(self.limit, spill_directory=self.directory)
        store._budget = self
        self.stores.append(store)
        return store

    def reserve(self, incoming: FitCounts, charge: int) -> None:
        while sum(s.retained_bytes for s in self.stores) + charge > self.limit:
            largest = max(self.stores, key=lambda s: s.retained_bytes)
            if not largest.retained_bytes:
                raise MemoryError('one fitting key exceeds fit_memory_limit')
            largest._spill()
            if incoming.spilled:
                return
        self.peak_retained_bytes = max(self.peak_retained_bytes,
                                       sum(s.retained_bytes for s in self.stores) + charge)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Attempt cleanup of every store even if one cleanup fails.
        from contextlib import ExitStack
        with ExitStack() as stack:
            for store in self.stores:
                stack.callback(store.close)


class NumericRuns:
    """Adapter for split_fit's sorted uint64 numeric runs; duplicate keys counted."""

    def __init__(self, budget: FitBudget):
        self.store = budget.counts()

    def append(self, values):
        self.store.add((int(v).to_bytes(8, 'big'), 1) for v in values)

    @property
    def size(self):
        return self.store.total

    def select(self, positions):
        import numpy as np
        return np.array([int.from_bytes(k, 'big') for k in
                         self.store.order_statistics(int(p) for p in positions)], dtype=np.uint64)


class ValueCounts:
    """Merge per-chunk typed vocabulary counts before constructing final arrays.

    Pickle only encodes internal keys; no external files or pickles are accepted.
    Final arrays scale with distinct values, not accumulated chunk observations.
    """

    def __init__(self, budget: FitBudget):
        self.store = budget.counts()
        self.dtype = None
        self.arrow = False

    def append(self, pair):
        import pickle

        import numpy as np
        values, counts = pair
        arrow = not isinstance(values, np.ndarray)
        dtype = values.type if arrow else values.dtype
        if self.dtype is not None and (self.dtype != dtype or self.arrow != arrow):
            raise ValueError('fitting count key type changed between chunks')
        self.dtype, self.arrow = dtype, arrow
        self.store.add((pickle.dumps(v.as_py() if arrow else v.item(), protocol=5), int(n))
                       for v, n in zip(values, counts, strict=True))

    def arrays(self):
        import pickle

        import numpy as np
        if self.arrow:
            import pyarrow as pa
            values = pa.array((pickle.loads(k) for k, _ in self.store.items()), type=self.dtype)
        else:
            values = np.fromiter((pickle.loads(k) for k, _ in self.store.items()), dtype=self.dtype)
        counts = np.fromiter((n for _, n in self.store.items()), dtype=np.int64)
        return values, counts

    def __bool__(self):
        return bool(self.store.total)


class TupleParts:
    """Count typed component tuples across chunks before GPU finalization."""

    def __init__(self, budget: FitBudget):
        self.store = budget.counts()
        self.types = None

    def append(self, part):
        import pickle

        import numpy as np
        missing, identities, counts = part
        types = [(False, a.dtype) if isinstance(a, np.ndarray) else (True, a.type) for a in identities]
        if self.types is not None and self.types != types:
            raise ValueError('fitting tuple type changed between chunks')
        self.types = types

        def records():
            for row, count in enumerate(counts):
                values = tuple((bool(m[row]), a[row].as_py() if arrow else a[row].item())
                               for m, a, (arrow, _) in zip(missing, identities, types, strict=True))
                # Disable memoization: identical tuples must have identical keys
                # regardless of whether equal text objects share Python identity.
                from io import BytesIO
                encoded = BytesIO()
                writer = pickle.Pickler(encoded, protocol=5)
                writer.fast = True
                writer.dump(values)
                yield encoded.getvalue(), int(count)
        self.store.add(records())

    def materialized(self):
        import pickle

        import numpy as np
        if self.types is None:
            return []
        missing, identities = [], []
        for i, (arrow, dtype) in enumerate(self.types):
            missing.append(np.fromiter((pickle.loads(k)[i][0] for k, _ in self.store.items()), dtype=bool))
            values = (pickle.loads(k)[i][1] for k, _ in self.store.items())
            if arrow:
                import pyarrow as pa
                identities.append(pa.array(values, type=dtype))
            else:
                identities.append(np.fromiter(values, dtype=dtype))
        counts = np.fromiter((n for _, n in self.store.items()), dtype=np.int64)
        return [(missing, identities, counts)]
