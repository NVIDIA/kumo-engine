# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a token cache, whichever `ColumnarTokenizer` built it.

A cache's part files are memory-mapped, or read with `os.pread` when there are more
of them than the process may keep open. A row
is a user and a range of that user's events, in `order_events` order. Batches
are padded the way collating a whole corpus with `collate_batch` and cutting it
with `minibatches` pads them; a contract test holds the two equal.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import resource
import warnings
from collections import OrderedDict
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, overload

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.kvt import EncodedKVTRecord, KeyValueTimeBatch
from tfm.contracts.schema import EntityProfile, TransactionSequence, ensure_utc
from tfm.contracts.tokenizers import (
    LAYOUT_KEY_VALUE_TIME,
    ColumnarTokenizer,
    FieldNamedTokenizer,
    ProfilePrefix,
    ProfilePrefixEncoder,
    event_array_problem,
)
from tfm.data.adapters.pragmatiq import load_profiles
from tfm.data.token_cache.build import ENGINE_ARRAYS, FORMAT_VERSION, part_header, train_ranges
from tfm.features.tokenizers.key_value_time import (
    PAD,
    collate_records,
    time_encode,
    time_encode_many,
)
from tfm.features.tokenizers.registry import tokenizer_class

_EMPTY = np.zeros(0, np.int64)


def _log_times(delta_us: NDArray[np.int64]) -> NDArray[np.float64]:
    """`time_encode` of microsecond gaps, float for float as the tokenizer computes it."""
    return time_encode_many(delta_us / 1e6)


def _ranges(lo: NDArray[np.int64], hi: NDArray[np.int64]) -> NDArray[np.int64]:
    """`concat(arange(lo[i], hi[i]))`."""
    n = hi - lo
    total = int(n.sum())
    if not total:
        return _EMPTY
    return np.repeat(lo - (np.cumsum(n) - n), n) + np.arange(total, dtype=np.int64)


def _sorted_keys(ids: pa.Array, path: Path) -> NDArray[np.bytes_]:
    """The user ids as fixed-width UTF-8 bytes, ascending, for binary search.

    Each id takes as many bytes as the longest one.
    """
    ids = ids.cast(pa.large_binary())
    offsets = np.frombuffer(ids.buffers()[1], np.int64)[ids.offset : ids.offset + len(ids) + 1]
    data = ids.buffers()[2]
    data = np.zeros(0, np.uint8) if data is None else np.frombuffer(data, np.uint8)
    data = data[offsets[0] : offsets[-1]]
    lengths = np.diff(offsets)
    if (data == 0).any():
        raise SchemaError(f"Token cache {path} has a user id containing a NUL byte.")
    width = max(int(lengths.max()) if len(lengths) else 1, 1)
    out = np.zeros((len(lengths), width), np.uint8)
    out[np.arange(width) < lengths[:, None]] = data
    keys = out.view(f"S{width}").ravel()
    if len(keys) > 1 and not (keys[1:] > keys[:-1]).all():
        raise SchemaError(f"Token cache {path} lists its users out of order; rebuild it.")
    return keys


_HEADER = len(part_header("0" * 32, 0))


def open_file_budget() -> int:
    """Files a cache may keep open: a quarter of this process's open-file limit."""
    soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if soft == resource.RLIM_INFINITY or soft < 0:
        return 1 << 20
    return max(1, soft // 4)


class CacheFiles:
    """A cache's part files, shared by its arrays and opened on first use.

    With no more files than the open-file budget (a quarter of the process's open-file
    limit), each file is memory-mapped once; otherwise read with `os.pread`, keeping at
    most the budget open.
    """

    def __init__(self, paths: Sequence[Path], build_id: str) -> None:
        self.paths, self.build_id = list(paths), build_id
        self.budget = open_file_budget()
        self._open: OrderedDict[int, int] = OrderedDict()
        self._maps: dict[int, np.ndarray] = {}
        self._checked: set[int] = set()

    def _check(self, p: int, head: bytes) -> None:
        """Refuse file `p` unless it opens with this cache's `part_header` for `p`."""
        if p not in self._checked:
            if head != part_header(self.build_id, p):
                raise SchemaError(
                    f"{self.paths[p]} is not part {p} of the cache whose manifest is beside "
                    "it (its header differs). Rebuild the cache with `tfm cache-tokens "
                    "--config ...` rather than copying part files between caches."
                )
            self._checked.add(p)

    @property
    def mapped(self) -> bool:
        return len(self.paths) <= self.budget

    def map(self, p: int) -> np.ndarray:
        """File `p` as bytes."""
        if p not in self._maps:
            m = np.memmap(self.paths[p], np.uint8, "r")
            self._check(p, bytes(m[:_HEADER]))
            self._maps[p] = m
        return self._maps[p]

    def fd(self, p: int) -> int:
        """File `p`'s open descriptor; the least recently used one closes past the budget."""
        if p in self._open:
            self._open.move_to_end(p)
            return self._open[p]
        fd = os.open(self.paths[p], os.O_RDONLY)
        self._checked.discard(p)  # a reopened path may name another file now
        try:
            self._check(p, os.pread(fd, _HEADER, 0))
        except BaseException:
            os.close(fd)
            raise
        self._open[p] = fd
        if len(self._open) > self.budget:
            os.close(self._open.popitem(last=False)[1])
        return self._open[p]

    def __getstate__(self) -> dict[str, Any]:
        return {"paths": self.paths, "build_id": self.build_id}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["paths"], state["build_id"])  # type: ignore[misc]

    def __del__(self, close: Callable[[int], None] = os.close) -> None:
        # `close` is bound at definition: at interpreter exit `os` may already be gone.
        for fd in getattr(self, "_open", {}).values():
            close(fd)


class ShardedArray:
    """One 1-D array stored as consecutive shards, each a run of bytes in a part file.

    Element `i` of shard `s` reads `files.paths[p][offset + i * itemsize]` plus `shift[s]`,
    where `pieces[s] == (p, offset)`. Indexing with an int, a slice or an integer array
    reads only the elements it asks for and returns a NumPy array (a scalar for an int).
    """

    def __init__(
        self,
        files: CacheFiles,
        pieces: Sequence[tuple[int, int]],
        sizes: Sequence[int],
        dtype: Any,
        *,
        shift: Sequence[int] | NDArray[np.int64] | None = None,
    ) -> None:
        self.files = files
        self.pieces = [(int(p), int(o)) for p, o in pieces]
        self.starts = np.concatenate([[0], np.cumsum(np.asarray(sizes, np.int64))])
        self._starts: list[int] = self.starts.tolist()
        self.shift = None if shift is None else np.asarray(shift, np.int64)
        self._stored = np.dtype(dtype)
        self.dtype = np.dtype(np.int64) if shift is not None else self._stored
        self.shape = (int(self.starts[-1]),)

    def _view(self, s: int, extra: int = 0) -> np.ndarray:
        """Shard `s` and the `extra` elements stored after it, memory-mapped, without
        its shift."""
        p, offset = self.pieces[s]
        n = (self._starts[s + 1] - self._starts[s] + extra) * self._stored.itemsize
        raw = self.files.map(p)[offset : offset + n]
        if len(raw) != n:
            raise SchemaError(f"{self.files.paths[p]} is shorter than its manifest says.")
        return raw.view(self._stored)

    def _read(self, s: int, a: int, b: int) -> np.ndarray:
        """Shard `s`'s elements `[a, b)`, with its shift."""
        if self.files.mapped:
            part = self._view(s)[a:b]
            return part if self.shift is None else part + self.shift[s]
        out = np.empty(b - a, self._stored)
        self._fill(s, memoryview(out).cast("B"), [a], [b - a])
        return out if self.shift is None else out + self.shift[s]

    def _fill(self, s: int, into: memoryview, starts: list[int], lengths: list[int]) -> None:
        """Read shard `s`'s runs `[start, start + length)` into `into`, back to back."""
        p, offset = self.pieces[s]
        fd = self.files.fd(p)
        size = self._stored.itemsize
        at = 0
        for start, length in zip(starts, lengths, strict=True):
            end, pos = at + length * size, offset + start * size
            while at < end:
                n = os.preadv(fd, [into[at:end]], pos)
                if not n:
                    raise SchemaError(f"{self.files.paths[p]} is shorter than its manifest says.")
                at, pos = at + n, pos + n

    def __len__(self) -> int:
        return self.shape[0]

    def shard(self, s: int) -> np.ndarray:
        """Shard `s`'s elements, without its shift."""
        values = self._read(s, 0, self._starts[s + 1] - self._starts[s])
        return values if self.shift is None else values - self.shift[s]

    def __getitem__(self, index: int | slice | NDArray[np.int64]) -> Any:
        if isinstance(index, (int, np.integer)):
            i = int(index) + len(self) if index < 0 else int(index)
            if not 0 <= i < len(self):
                raise IndexError(f"index {int(index)} is out of range for length {len(self)}")
            s = bisect.bisect_right(self._starts, i) - 1
            return self._read(s, i - self._starts[s], i - self._starts[s] + 1)[0]
        if isinstance(index, slice):
            a, b, step = index.indices(len(self))
            if step == 1:
                return self._span(a, max(a, b))
            index = np.arange(a, b, step, dtype=np.int64)
        idx = np.asarray(index, np.int64)
        return self.gather(self.group(idx), idx.shape)

    def _span(self, a: int, b: int) -> np.ndarray:
        """Elements `[a, b)`, read shard by shard."""
        out = np.empty(b - a, self.dtype)
        first = a
        while a < b:
            s = bisect.bisect_right(self._starts, a) - 1
            stop = min(b, self._starts[s + 1])
            out[a - first : stop - first] = self._read(
                s, a - self._starts[s], stop - self._starts[s]
            )
            a = stop
        return out

    def group(self, idx: NDArray[np.int64]) -> list[tuple[int, Any, NDArray[np.int64]]]:
        """`(shard, positions in idx, indices within the shard)` for each shard idx touches.

        `gather` reads them; arrays whose shards hold the same numbers of elements can
        share one grouping.
        """
        if idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError(f"an index is out of range for length {len(self)}")
        if len(self.pieces) == 1:
            return [(0, None, idx)]
        flat = idx.ravel()
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        narrow = shard.astype(np.uint16) if len(self.pieces) <= 1 << 16 else shard
        order = np.argsort(narrow, kind="stable")  # a radix sort for uint16
        bounds = np.concatenate([[0], np.cumsum(np.bincount(shard, minlength=len(self.pieces)))])
        groups = []
        for s in np.flatnonzero(np.diff(bounds)).tolist():
            take = order[bounds[s] : bounds[s + 1]]
            groups.append((s, take, flat[take] - self._starts[s]))
        return groups

    def gather(
        self,
        groups: list[tuple[int, Any, NDArray[np.int64]]],
        shape: tuple[int, ...],
        step: int = 0,
    ) -> np.ndarray:
        """The elements `group` located, each moved `step` places within its shard's
        stored part, which may run past the shard (a block's terminal token offset).

        Each shard's indices are read as their contiguous runs; when they lie close
        together, as one read of the span they cover.
        """
        out = np.empty(shape, self.dtype)
        flat_out = out.reshape(-1)
        for s, take, local in groups:
            local = local.ravel() + step
            if not len(local):
                continue
            target = slice(None) if take is None else take
            if self.files.mapped:
                # The last shard's logical size already holds its terminal element.
                part = self._view(s, step if s + 1 < len(self.pieces) else 0)[local]
                flat_out[target] = part if self.shift is None else part + self.shift[s]
                continue
            lo, hi = int(local.min()), int(local.max()) + 1
            if hi - lo <= 4 * len(local) + 4096:
                flat_out[target] = self._read(s, lo, hi)[local - lo]
                continue
            begins = np.concatenate([[0], np.flatnonzero(np.diff(local) != 1) + 1])
            lengths = np.diff(np.append(begins, len(local)))
            values = np.empty(len(local), self._stored)
            self._fill(s, memoryview(values).cast("B"), local[begins].tolist(), lengths.tolist())
            flat_out[target] = values if self.shift is None else values + self.shift[s]
        return out

    def _shards(self, dtype: Any) -> Iterator[tuple[int, int, np.ndarray]]:
        """`(start, stop, values)` of each shard in turn, as `dtype`."""
        for s in range(len(self.pieces)):
            a, b = self._starts[s], self._starts[s + 1]
            yield a, b, np.array(self._read(s, 0, b - a), dtype)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        if copy is False:
            raise ValueError("A ShardedArray is read into a new array; it cannot avoid a copy.")
        out = np.empty(len(self), self.dtype if dtype is None else dtype)
        for a, b, values in self._shards(out.dtype):
            out[a:b] = values
        return out

    def to_tensor(self, device: torch.device, dtype: Any = None) -> torch.Tensor:
        """The whole array on `device`, copied from the host one shard at a time."""
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        out = torch.empty(
            len(self), dtype=torch.from_numpy(np.zeros(0, dtype)).dtype, device=device
        )
        for a, b, values in self._shards(dtype):
            out[a:b] = torch.from_numpy(values).to(device)
        return out


@dataclass(frozen=True)
class CacheRows:
    """Sequences as `(user, [lo, hi))` event ranges, all cut at one cutoff."""

    user: NDArray[np.int64]
    lo: NDArray[np.int64]
    hi: NDArray[np.int64]
    cutoff: datetime

    def __len__(self) -> int:
        return len(self.user)

    def take(self, index: Sequence[int] | NDArray[np.int64]) -> CacheRows:
        idx = np.asarray(index, np.int64)
        return CacheRows(self.user[idx], self.lo[idx], self.hi[idx], self.cutoff)


_TOKEN_ARRAYS = ("token_offsets", "key_ids", "value_ids", "positions")


class TokenCache:
    """A built cache: its manifest, fitted tokenizer state, and arrays read from its shards."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise SchemaError(
                f"No token cache at {self.path}; build it with `tfm cache-tokens --config ...`."
            )
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != FORMAT_VERSION:
            raise SchemaError(
                f"Token cache {self.path} has format {self.manifest.get('format_version')}; "
                f"this code reads format {FORMAT_VERSION}. Rebuild it with "
                "`tfm cache-tokens --config ...`."
            )
        self.tokenizer_state: dict[str, Any] = json.loads(
            (self.path / "tokenizer.json").read_text(encoding="utf-8")
        )
        users = pq.read_table(self.path / "users.parquet")
        self._user_keys = _sorted_keys(users.column("user_id").combine_chunks(), self.path)
        self.user_start = users.column("event_start").to_numpy().astype(np.int64)
        self.user_stop = users.column("event_stop").to_numpy().astype(np.int64)

        blocks = self.manifest["blocks"]
        dtypes = self.manifest["arrays"]
        invalid_events = {
            name: problem
            for name, dtype in dtypes.items()
            if name not in (*_TOKEN_ARRAYS, *ENGINE_ARRAYS)
            and (problem := event_array_problem(np.dtype(dtype)))
        }
        if invalid_events:
            raise SchemaError(
                f"Per-event cache arrays need indexable integer dtypes: {invalid_events}. "
                "Rebuild the cache with an integer-valued tokenizer."
            )
        files = CacheFiles(
            [self.path / "parts" / f"{p:05d}.bin" for p in range(self.manifest["parts"])],
            self.manifest["build_id"],
        )
        if not files.mapped:
            n = len(files.paths)
            warnings.warn(
                f"Token cache {self.path} has {n} files, more than the {files.budget} this "
                "process may keep open (a quarter of its open-file limit), so it reads them "
                f"with os.pread and reopens files between batches. Raise the limit to {4 * n} "
                f"or more (`ulimit -n {4 * n}`) to memory-map them.",
                stacklevel=2,
            )
        events = [int(b["events"]) for b in blocks]
        tokens = [int(b["tokens"]) for b in blocks]

        def load(name: str, sizes: list[int] | None = None, **kw: Any) -> ShardedArray:
            if sizes is None:
                sizes = tokens if name in _TOKEN_ARRAYS[1:] else events
            pieces = [(b["part"], b["offsets"][name]) for b in blocks]
            return ShardedArray(files, pieces, sizes, dtypes[name], **kw)

        # Each block's offsets start at 0; the last block's final one is the total.
        self.token_offsets = load(
            "token_offsets",
            [*events[:-1], events[-1] + 1],
            shift=np.concatenate([[0], np.cumsum(tokens)[:-1]]),
        )
        self.ts_us = load("ts_us")
        self.id_us = load("id_us")
        self.events = {
            name: load(name) for name in dtypes if name not in (*_TOKEN_ARRAYS, *ENGINE_ARRAYS)
        }
        """The tokenizer's per-event arrays by name, in the order it emits them."""
        self.amount = load("amount")
        self.by_loader = load("by_loader")
        self.key_ids = load("key_ids")
        self.value_ids = load("value_ids")
        self.positions = load("positions")

    @property
    def n_users(self) -> int:
        return len(self._user_keys)

    def user_ids(self, index: Sequence[int] | NDArray[np.int64] | None = None) -> list[str]:
        """The ids of the users at `index`, or of every cached user."""
        keys = self._user_keys if index is None else self._user_keys[np.asarray(index, np.int64)]
        return [k.decode() for k in keys.tolist()]

    def user_index(self, users: Sequence[str]) -> NDArray[np.int64]:
        """Each user's position in the cache, or -1 for a user the cache does not hold."""
        if not len(users):
            return _EMPTY
        keys = np.array([u.encode() for u in users])
        where = np.minimum(np.searchsorted(self._user_keys, keys), max(self.n_users - 1, 0))
        found = (self.n_users > 0) & (self._user_keys[where] == keys)
        return np.where(found, where, -1).astype(np.int64)

    def check_identity(self, expected: Mapping[str, Any]) -> None:
        """Raise unless the cache was built from the book, users, fit, tokenizer, and amount
        field expected."""
        for name, want in expected.items():
            have = self.manifest.get(name)
            if have != want:
                raise SchemaError(
                    f"Token cache {self.path} was built with {name} {have!r}, but this run "
                    f"needs {want!r}. Rebuild it with `tfm cache-tokens`."
                )

    def profiles(self) -> dict[str, EntityProfile]:
        """The cached users' profiles, as the book held them when the cache was built."""
        path = self.path / "profiles.parquet"
        if not path.exists():
            return {}
        return {p.entity_id: p for p in load_profiles(pq.read_table(path))}

    def check_event_arrays(self, names: Sequence[str]) -> None:
        """Raise unless the cache holds every per-event array in `names`."""
        if missing := [name for name in names if name not in self.events]:
            raise UnsupportedCombinationError(
                f"The model reads the per-event arrays {missing}, which token cache "
                f"{self.path} does not hold (it holds {sorted(self.events)}). Use a tokenizer "
                "that emits them and rebuild the cache, or a model that does not read them."
            )

    def tokenizer(
        self,
        profiles: Mapping[str, EntityProfile] | None = None,
        *,
        max_events: int | None,
    ) -> CachedTokenizer:
        """The tokenizer that built the cache, fitted from its state, reading tokens from
        the cache; `profiles` defaults to the cache's own. Each row keeps its last
        `max_events` events."""
        cls = tokenizer_class(self.manifest["tokenizer"])
        fitted: ColumnarTokenizer[Any] = cls.from_state(self.tokenizer_state)
        encoder: object = fitted
        if isinstance(encoder, ProfilePrefixEncoder):
            encoder.set_profiles(self.profiles() if profiles is None else profiles)
        return CachedTokenizer(self, fitted, max_events=max_events)

    # Rows -----------------------------------------------------------------

    def train_rows(
        self,
        *,
        train_end: datetime,
        lookback: timedelta | None,
        max_events: int,
        users: Sequence[str] | None = None,
    ) -> CacheRows:
        """`build_entity_sequences` over train events (`ts < train_end`) at `train_end`."""
        index = self._users(users)
        lo, hi = self._train_ranges(index, train_end, lookback, max_events)
        keep = hi > lo
        return CacheRows(index[keep], lo[keep], hi[keep], ensure_utc(train_end))

    def train_windows(
        self,
        *,
        train_end: datetime,
        lookback: timedelta | None,
        window_events: int,
        stride: int | None = None,
        users: Sequence[str] | None = None,
    ) -> CacheRows:
        """`build_entity_windows` over train events at `train_end`."""
        if window_events <= 0:
            raise ValueError(f"window_events must be positive, got {window_events}.")
        step = window_events if stride is None else stride
        if step <= 0:
            raise ValueError(f"stride must be positive, got {step}.")
        index = self._users(users)
        lo, hi = self._train_ranges(index, train_end, lookback, 0)
        out_user: list[NDArray[np.int64]] = []
        out_lo: list[NDArray[np.int64]] = []
        out_hi: list[NDArray[np.int64]] = []
        for u, a, b in zip(index.tolist(), lo.tolist(), hi.tolist(), strict=True):
            n = b - a
            if n <= 0:
                continue
            # Windows start every `step` events; the first one to reach the end is the last.
            last = 0 if n <= window_events else -(-(n - window_events) // step)
            starts = step * np.arange(last + 1, dtype=np.int64)
            starts = a + starts[starts < n]
            out_user.append(np.full(len(starts), u, np.int64))
            out_lo.append(starts)
            out_hi.append(np.minimum(starts + window_events, b))
        if not out_user:
            return CacheRows(_EMPTY, _EMPTY, _EMPTY, ensure_utc(train_end))
        return CacheRows(
            np.concatenate(out_user),
            np.concatenate(out_lo),
            np.concatenate(out_hi),
            ensure_utc(train_end),
        )

    def _users(self, users: Sequence[str] | None) -> NDArray[np.int64]:
        if users is None:
            return np.arange(self.n_users, dtype=np.int64)
        index = self.user_index(users)
        if (index < 0).any():
            missing = [u for u, i in zip(users, index.tolist(), strict=True) if i < 0]
            raise SchemaError(
                f"{len(missing)} users are not in token cache {self.path} (first: {missing[0]!r})."
            )
        return np.unique(index)

    def _train_ranges(
        self,
        index: NDArray[np.int64],
        train_end: datetime,
        lookback: timedelta | None,
        max_events: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        lo = np.empty(self.n_users, np.int64)
        hi = np.empty(self.n_users, np.int64)
        # A shard holds whole users, so each shard's users are ranged from that shard alone.
        bounds = np.searchsorted(self.user_start, self.ts_us.starts)
        for s, (a, b) in enumerate(zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True)):
            if b == a:
                continue
            first = int(self.ts_us.starts[s])
            offsets = np.concatenate([self.user_start[a:b], self.user_stop[b - 1 : b]]) - first
            shard_lo, shard_hi = train_ranges(
                np.asarray(self.ts_us.shard(s)),
                offsets,
                train_end=train_end,
                lookback=lookback,
                max_events=max_events,
            )
            lo[a:b], hi[a:b] = shard_lo + first, shard_hi + first
        return lo[index], hi[index]

    # Events ---------------------------------------------------------------

    def event_position(self, event_id: str) -> int:
        """The cache position of a pragmatiq `event_id` (`user:microseconds:index`)."""
        try:
            user, micros, index = event_id.rsplit(":", 2)
            loader, id_us = int(index), int(micros)
        except ValueError as err:
            raise SchemaError(
                f"Event {event_id!r} is not in token cache {self.path}: the cache finds an event "
                "by its pragmatiq id (`user:microseconds:index`), so a book with other event ids "
                "cannot use data.token_cache. Alias a copied event to its original."
            ) from err
        key = user.encode()
        u = int(np.searchsorted(self._user_keys, key))
        if u >= self.n_users or self._user_keys[u] != key:
            raise SchemaError(
                f"Event {event_id!r} belongs to a user outside token cache {self.path}."
            )
        a, b = int(self.user_start[u]), int(self.user_stop[u])
        if not 0 <= loader < b - a:
            raise SchemaError(f"Event {event_id!r} is not in token cache {self.path}.")
        position = a + int(self.by_loader[a + loader])
        if int(self.id_us[position]) != id_us:
            raise SchemaError(
                f"Event {event_id!r} does not match token cache {self.path}; the book changed."
            )
        return position

    # Batches --------------------------------------------------------------

    def capped(self, rows: CacheRows, max_events: int | None) -> NDArray[np.int64]:
        """Each row's first encoded event: a row keeps its last `max_events` (all of them for
        `None` or a value below 1, as in `build_entity_sequences`)."""
        if max_events is None or max_events < 1:
            return rows.lo
        return np.maximum(rows.lo, rows.hi - max_events)

    def corpus_shape(
        self, rows: CacheRows, tokenizer: CachedTokenizer, max_length: int
    ) -> tuple[int, int, int]:
        """`(events, tokens, profile tokens)` one `collate` over all of `rows` pads to."""
        if not len(rows):
            return 1, 1, 1
        lo = self.capped(rows, tokenizer.max_events)
        max_events = max(int((rows.hi - lo).max()), 1)
        max_tok = 1
        ends = np.cumsum(rows.hi - lo)
        start = 0
        while start < len(lo):  # about 1M events at a time
            stop = max(int(np.searchsorted(ends, ends[start] + (1 << 20), side="left")), start + 1)
            events = _ranges(lo[start:stop], rows.hi[start:stop])
            if len(events):
                counts = self.token_offsets[events + 1] - self.token_offsets[events]
                max_tok = max(max_tok, int(counts.max()))
            start = stop
        max_prof = max(
            max(
                len(tokenizer.encode_profile(user, rows.cutoff)[0])
                for user in self.user_ids(np.unique(rows.user))
            ),
            1,
        )
        return max_events, max(min(max_tok, max_length), 1), max_prof

    def collate(
        self,
        rows: CacheRows,
        tokenizer: CachedTokenizer,
        *,
        max_length: int,
        pad_to: tuple[int, int, int] | None = None,
        event_arrays: Collection[str] | None = None,
    ) -> KeyValueTimeBatch:
        """`collate_batch` for cached rows, without a label window.

        `pad_to` is `corpus_shape(...)` of a larger corpus these rows are cut
        from. `event_arrays` names the per-event arrays to gather (default: all).
        Fraud labels are zero: the cache holds no labels. Probes read theirs from the book,
        and no pretraining objective a key-value-time stack accepts reads them.
        """
        lo, hi = self.capped(rows, tokenizer.max_events), rows.hi
        n_ev = hi - lo
        bsz = len(rows)
        max_events, max_tok, max_prof = (
            self.corpus_shape(rows, tokenizer, max_length) if pad_to is None else pad_to
        )

        event_index = _ranges(lo, hi)
        row_of_event = np.repeat(np.arange(bsz, dtype=np.int64), n_ev)
        slot_of_event = np.arange(len(event_index), dtype=np.int64) - np.repeat(
            np.cumsum(n_ev) - n_ev, n_ev
        )
        # One shard grouping per index array, shared by every array of that length.
        events = self.id_us.group(event_index)
        tok_lo = self.token_offsets.gather(events, event_index.shape)
        tok_n = self.token_offsets.gather(events, event_index.shape, step=1) - tok_lo
        take = np.minimum(tok_n, max_tok)
        tok_index = _ranges(tok_lo, tok_lo + take)
        tokens = self.key_ids.group(tok_index)
        tok_where = (
            torch.from_numpy(np.repeat(row_of_event, take)),
            torch.from_numpy(np.repeat(slot_of_event, take)),
            torch.from_numpy(
                np.arange(len(tok_index), dtype=np.int64) - np.repeat(np.cumsum(take) - take, take)
            ),
        )
        shape = (bsz, max_events, max_tok)
        key_ids = torch.full(shape, PAD, dtype=torch.long)
        value_ids = torch.full(shape, PAD, dtype=torch.long)
        positions = torch.zeros(shape, dtype=torch.long)
        token_mask = torch.zeros(shape, dtype=torch.long)
        for target, source in (
            (key_ids, self.key_ids),
            (value_ids, self.value_ids),
            (positions, self.positions),
        ):
            values = source.gather(tokens, tok_index.shape)
            target[tok_where] = torch.from_numpy(np.asarray(values, np.int64))
        token_mask[tok_where] = 1

        ev_where = (torch.from_numpy(row_of_event), torch.from_numpy(slot_of_event))
        event_mask = torch.zeros((bsz, max_events), dtype=torch.long)
        event_mask[ev_where] = 1
        id_us = np.asarray(self.id_us.gather(events, event_index.shape), np.int64)
        time_log = torch.zeros((bsz, max_events), dtype=torch.float32)
        elapsed_log = torch.zeros((bsz, max_events), dtype=torch.float32)
        if len(id_us):
            ends = np.cumsum(n_ev)[row_of_event]
            last = id_us[ends - 1]
            first = id_us[ends - n_ev[row_of_event]]
            time_log[ev_where] = torch.from_numpy(_log_times(last - id_us).astype(np.float32))
            elapsed_log[ev_where] = torch.from_numpy(_log_times(id_us - first).astype(np.float32))
        per_event: dict[str, torch.Tensor] = {}
        for name, source in self.events.items():
            if event_arrays is not None and name not in event_arrays:
                continue
            tensor = torch.zeros((bsz, max_events), dtype=torch.long)
            values = source.gather(events, event_index.shape)
            tensor[ev_where] = torch.from_numpy(np.asarray(values, np.int64))
            per_event[name] = tensor

        entity_ids = tuple(self.user_ids(rows.user))
        # `TransactionSequence`'s default id.
        cutoff = ensure_utc(rows.cutoff).isoformat()
        return KeyValueTimeBatch(
            key_ids=key_ids,
            value_ids=value_ids,
            positions=positions,
            token_mask=token_mask,
            event_mask=event_mask,
            time_log=time_log,
            elapsed_log=elapsed_log,
            events=per_event,
            **profile_tensors(tokenizer, entity_ids, rows.cutoff, max_prof),
            entity_ids=entity_ids,
            sequence_ids=tuple(f"{e}:{cutoff}" for e in entity_ids),
            fraud_labels=torch.zeros(bsz, dtype=torch.float32),
            numeric_features=self.raw_features(rows.lo, rows.hi),
        )

    def raw_features(self, lo: NDArray[np.int64], hi: NDArray[np.int64]) -> torch.Tensor:
        """`raw_feature_vector` of each whole row `[lo, hi)` (not capped): float32
        `[len(lo), 5]`, zeros for an empty row. The last event's UTC hour and weekday come
        from `ts_us`, so they need no per-event array.

        Absent/non-finite amounts contribute zero, without modifying stored arrays.
        The amounts are summed with Python's `sum` (Neumaier compensation, Python 3.12+) and
        maximised with `max`, in event order, so every value equals the object path's.
        Rows up to `stop` events are summed together, one step per event position; each
        longer row is summed on its own. `stop` minimises steps plus single rows, taking
        one step and one single row to cost about the same.
        """
        out = np.zeros((len(lo), 5), np.float64)
        rows = np.flatnonzero(hi > lo)
        if len(rows):
            first, n = lo[rows], hi[rows] - lo[rows]
            lengths = np.sort(n)
            steps = np.arange(int(lengths[-1]) + 1)
            longer = len(lengths) - np.searchsorted(lengths, steps, side="right")
            stop = max(int(np.argmin(steps + longer)), 1)
            x = np.asarray(self.amount[first], np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            total, comp, top = 0.0 + x, np.zeros(len(rows)), x.copy()
            k = np.flatnonzero(n <= stop)
            for j in range(1, stop):
                k = k[n[k] > j]
                x = np.asarray(self.amount[first[k] + j], np.float64)
                x = np.where(np.isfinite(x), x, 0.0)
                f = total[k]
                t = f + x
                comp[k] += np.where(np.abs(f) >= np.abs(x), (f - t) + x, (x - t) + f)
                total[k] = t
                top[k] = np.where(x > top[k], x, top[k])
            done = (comp != 0) & np.isfinite(comp)
            total[done] += comp[done]
            for i in np.flatnonzero(n > stop).tolist():
                values = np.asarray(self.amount[first[i] : first[i] + n[i]], np.float64)
                amounts = np.where(np.isfinite(values), values, 0.0).tolist()
                total[i], top[i] = sum(amounts), max(amounts)
            last_us = np.asarray(self.ts_us[hi[rows] - 1], np.int64)
            hour = np.floor_divide(last_us, 3_600_000_000) % 24
            dow = (np.floor_divide(last_us, 86_400_000_000) + 3) % 7
            out[rows] = np.stack([total / n, top, n, hour, dow], axis=1)
        return torch.from_numpy(out.astype(np.float32))


def profile_tensors(
    tokenizer: CachedTokenizer, entities: Sequence[str], cutoff: datetime, max_prof: int
) -> dict[str, torch.Tensor]:
    """Profile prefixes, padded as `collate_records` pads them.

    Each tensor is `[len(entities), max_prof]`: `prof_key_ids`, `prof_value_ids`,
    `prof_positions` and `prof_mask` int64, `prof_time_log` float32.
    """
    bsz = len(entities)
    prof_key = torch.full((bsz, max_prof), PAD, dtype=torch.long)
    prof_val = torch.full((bsz, max_prof), PAD, dtype=torch.long)
    prof_pos = torch.zeros((bsz, max_prof), dtype=torch.long)
    prof_mask = torch.zeros((bsz, max_prof), dtype=torch.long)
    prof_time = torch.zeros((bsz, max_prof), dtype=torch.float32)
    for b, entity in enumerate(entities):
        keys, values, positions, offsets, times = tokenizer.encode_profile(entity, cutoff)
        n_prof = min(len(keys), max_prof)
        if not n_prof:
            continue
        prof_key[b, :n_prof] = torch.tensor(keys[:n_prof], dtype=torch.long)
        prof_val[b, :n_prof] = torch.tensor(values[:n_prof], dtype=torch.long)
        prof_pos[b, :n_prof] = torch.tensor(positions[:n_prof], dtype=torch.long)
        prof_mask[b, :n_prof] = 1
        if times:
            expanded = [0.0] * n_prof
            for i, tval in enumerate(times):
                for j in range(offsets[i], min(offsets[i + 1], n_prof)):
                    expanded[j] = tval
            prof_time[b, :n_prof] = torch.tensor(expanded, dtype=torch.float32)
    return {
        "prof_key_ids": prof_key,
        "prof_value_ids": prof_val,
        "prof_positions": prof_pos,
        "prof_mask": prof_mask,
        "prof_time_log": prof_time,
    }


class CachedBatches(Sequence[KeyValueTimeBatch]):
    """Minibatches of cached rows, collated when indexed.

    The same rows, in the same order and padding, as collating all of `rows`,
    `shuffled_rows(batch, shuffle_seed)`, then `minibatches(batch, batch_size)`.
    Each batch carries the per-event arrays `event_arrays` (default: all).
    """

    def __init__(
        self,
        cache: TokenCache,
        rows: CacheRows,
        tokenizer: CachedTokenizer,
        *,
        batch_size: int,
        max_length: int,
        shuffle_seed: int | None,
        event_arrays: Collection[str] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if event_arrays is not None:
            cache.check_event_arrays(tuple(event_arrays))
        self.event_arrays = event_arrays
        self.cache, self.rows, self.tokenizer = cache, rows, tokenizer
        self.batch_size, self.max_length = batch_size, max_length
        if shuffle_seed is None:
            self.order = np.arange(len(rows), dtype=np.int64)
        else:
            generator = torch.Generator().manual_seed(shuffle_seed)
            self.order = torch.randperm(len(rows), generator=generator).numpy().astype(np.int64)
        self.pad_to = cache.corpus_shape(rows, tokenizer, max_length)

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    @overload
    def __getitem__(self, index: int) -> KeyValueTimeBatch: ...
    @overload
    def __getitem__(self, index: slice) -> list[KeyValueTimeBatch]: ...
    def __getitem__(self, index: int | slice) -> KeyValueTimeBatch | list[KeyValueTimeBatch]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._collate(slice(index * self.batch_size, (index + 1) * self.batch_size))

    def _collate(self, batch: slice) -> KeyValueTimeBatch:
        return self.cache.collate(
            self.rows.take(self.order[batch]),
            self.tokenizer,
            max_length=self.max_length,
            pad_to=self.pad_to,
            event_arrays=self.event_arrays,
        )


class CachedTokenizer:
    """The tokenizer that built a cache, reading each event's tokens and per-event arrays
    from the cache by `event_id` instead of encoding them.

    A sequence may hold any cached events, each row keeps its last `max_events`, and a
    profile prefix comes from the tokenizer when it is a `ProfilePrefixEncoder`. An event the
    cache does not hold, such as a splice's copy, needs an alias to the event it was
    copied from (`add_aliases`).
    """

    layout: str = LAYOUT_KEY_VALUE_TIME
    support_tier: str = "experimental"

    def __init__(
        self, cache: TokenCache, tokenizer: ColumnarTokenizer[Any], *, max_events: int | None
    ) -> None:
        self.cache = cache
        self.tokenizer = tokenizer
        self.max_events = max_events
        self.name: str = cache.manifest["tokenizer"]
        """The import path of the tokenizer that built the cache."""
        self._aliases: dict[str, str] = {}

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def add_aliases(self, aliases: Mapping[str, str]) -> None:
        """Event ids that stand for other cached events."""
        self._aliases.update(aliases)

    def fit(self, sequences: Sequence[TransactionSequence]) -> CachedTokenizer:
        raise UnsupportedCombinationError(
            "A cached tokenizer was fit when its cache was built; it is not refit per run."
        )

    def encode_profile(self, entity_id: str, cutoff: datetime) -> ProfilePrefix:
        """The tokenizer's profile prefix, or none when it is not a `ProfilePrefixEncoder`."""
        if isinstance(self.tokenizer, ProfilePrefixEncoder):
            return self.tokenizer.encode_profile(entity_id, cutoff)
        return [], [], [], [0], []

    def field_names(self) -> dict[int, str]:
        """The tokenizer's field names, or none when it is not a `FieldNamedTokenizer`."""
        if isinstance(self.tokenizer, FieldNamedTokenizer):
            return self.tokenizer.field_names()
        return {}

    def encode(self, sequence: TransactionSequence) -> EncodedKVTRecord:
        events = list(sequence.events)
        cap = self.max_events
        if cap is not None and 0 < cap < len(events):
            events = events[-cap:]
        cache = self.cache
        where = np.asarray(
            [cache.event_position(self._aliases.get(e.event_id, e.event_id)) for e in events],
            np.int64,
        )
        tok_lo = np.asarray(cache.token_offsets[where], np.int64)
        tok_hi = np.asarray(cache.token_offsets[where + 1], np.int64)
        index = _ranges(tok_lo, tok_hi)
        # A copied event keeps its own time; its tokens are the donor event's.
        ts_us = [int(ensure_utc(e.event_timestamp).timestamp() * 1_000_000) for e in events]
        tlog = tuple(time_encode(max(ts_us[-1] - ts, 0) / 1e6) for ts in ts_us) if ts_us else ()
        prof_k, prof_v, prof_p, prof_off, prof_t = self.encode_profile(
            sequence.entity_id, sequence.cutoff_time
        )
        return EncodedKVTRecord(
            entity_id=sequence.entity_id,
            sequence_id=sequence.sequence_id or sequence.entity_id,
            key_ids=tuple(np.asarray(cache.key_ids[index], np.int64).tolist()),
            value_ids=tuple(np.asarray(cache.value_ids[index], np.int64).tolist()),
            positions=tuple(np.asarray(cache.positions[index], np.int64).tolist()),
            event_offsets=(0, *np.cumsum(tok_hi - tok_lo).tolist()),
            event_ts_us=tuple(ts_us),
            time_log=tlog,
            events={
                name: tuple(np.asarray(array[where], np.int64).tolist())
                for name, array in cache.events.items()
            },
            prof_key_ids=tuple(prof_k),
            prof_value_ids=tuple(prof_v),
            prof_positions=tuple(prof_p),
            prof_offsets=tuple(prof_off),
            prof_time_log=tuple(prof_t),
        )

    def collate(
        self, sequences: Sequence[TransactionSequence], *, max_length: int
    ) -> KeyValueTimeBatch:
        """Pad records into a batch (`collate_records`)."""
        return collate_records(
            sequences, [self.encode(seq) for seq in sequences], max_length=max_length
        )

    def manifest(self) -> dict[str, object]:
        """The tokenizer as the cache records it; its fitted state is the cache's
        `tokenizer.json`."""
        return {
            "tokenizer": self.name,
            "layout": self.layout,
            "support_tier": self.support_tier,
            "vocab_size": self.vocab_size,
            "tokenizer_settings": self.tokenizer.settings(),
            "max_events": self.max_events,
            "event_arrays": list(self.cache.events),
            "token_cache": str(self.cache.path),
        }


__all__ = [
    "CacheRows",
    "CachedBatches",
    "CachedTokenizer",
    "CacheFiles",
    "open_file_budget",
    "ShardedArray",
    "TokenCache",
    "profile_tensors",
]
