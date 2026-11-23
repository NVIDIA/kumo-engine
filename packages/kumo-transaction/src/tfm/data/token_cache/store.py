# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a key-value-time token cache (ADR 0016).

Arrays are memory-mapped, so a run holds only the rows it is batching. A row
is a user and a range of that user's events, in `order_events` order. Batches
are padded the way collating a whole corpus with `collate_batch` and cutting it
with `minibatches` pads them; a contract test holds the two equal.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, overload

import numpy as np
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.kvt import EncodedKVTRecord, KeyValueTimeBatch
from tfm.contracts.schema import EntityProfile, TransactionSequence, ensure_utc
from tfm.data.adapters.pragmatiq import load_profiles
from tfm.data.token_cache.build import FORMAT_VERSION, train_ranges
from tfm.features.tokenizers.key_value_time import PAD, KeyValueTimeTokenizer, time_encode

_EMPTY = np.zeros(0, np.int64)


def _log_times(delta_us: NDArray[np.int64]) -> list[float]:
    """`time_encode` of microsecond gaps, float for float as the tokenizer computes it."""
    return [time_encode(d / 1e6) for d in delta_us.tolist()]


def _ranges(lo: NDArray[np.int64], hi: NDArray[np.int64]) -> NDArray[np.int64]:
    """`concat(arange(lo[i], hi[i]))`."""
    n = hi - lo
    total = int(n.sum())
    if not total:
        return _EMPTY
    return np.repeat(lo - (np.cumsum(n) - n), n) + np.arange(total, dtype=np.int64)


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


class TokenCache:
    """A built cache: its manifest, fitted tokenizer state, and memory-mapped arrays."""

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
                f"this code reads format {FORMAT_VERSION}. Rebuild it."
            )
        self.tokenizer_state: dict[str, Any] = json.loads(
            (self.path / "tokenizer.json").read_text(encoding="utf-8")
        )
        users = pq.read_table(self.path / "users.parquet")
        self.user_ids: tuple[str, ...] = tuple(str(u) for u in users.column("user_id").to_pylist())
        self.user_start = users.column("event_start").to_numpy().astype(np.int64)
        self.user_stop = users.column("event_stop").to_numpy().astype(np.int64)
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}

        def load(name: str) -> np.ndarray:
            return np.load(self.path / f"{name}.npy", mmap_mode="r")

        self.ts_us = load("ts_us")
        self.id_us = load("id_us")
        self.hour = load("hour")
        self.dow = load("dow")
        self.dom = load("dom")
        self.amount = load("amount")
        self.by_loader = load("by_loader")
        self.token_offsets = load("token_offsets")
        self.key_ids = load("key_ids")
        self.value_ids = load("value_ids")
        self.positions = load("positions")

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    def check_identity(self, expected: Mapping[str, Any]) -> None:
        """Raise unless the cache was built from the book, users, fit, and tokenizer expected."""
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

    def tokenizer(
        self, profiles: Mapping[str, EntityProfile] | None = None
    ) -> CachedKeyValueTimeTokenizer:
        """The fitted tokenizer; `profiles` defaults to the cache's own."""
        return CachedKeyValueTimeTokenizer.from_cache(
            self, self.profiles() if profiles is None else profiles
        )

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
        missing = [u for u in users if u not in self.user_index]
        if missing:
            raise SchemaError(
                f"{len(missing)} users are not in token cache {self.path} (first: {missing[0]!r})."
            )
        return np.asarray(sorted({self.user_index[u] for u in users}), np.int64)

    def _train_ranges(
        self,
        index: NDArray[np.int64],
        train_end: datetime,
        lookback: timedelta | None,
        max_events: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        offsets = np.concatenate([self.user_start, self.user_stop[-1:]]).astype(np.int64)
        lo, hi = train_ranges(
            np.asarray(self.ts_us),
            offsets,
            train_end=train_end,
            lookback=lookback,
            max_events=max_events,
        )
        return lo[index], hi[index]

    # Events ---------------------------------------------------------------

    def event_position(self, event_id: str) -> int:
        """The cache position of a pragmatiq `event_id` (`user:microseconds:index`)."""
        try:
            user, micros, index = event_id.rsplit(":", 2)
            loader, id_us = int(index), int(micros)
        except ValueError as err:
            raise SchemaError(
                f"Event {event_id!r} is not in token cache {self.path}: it is not a pragmatiq "
                "event id (`user:microseconds:index`). Alias a copied event to its original."
            ) from err
        u = self.user_index.get(user)
        if u is None:
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

    def capped(self, rows: CacheRows, tokenizer: KeyValueTimeTokenizer) -> NDArray[np.int64]:
        """Each row's first encoded event: `encode` keeps the last `max_events_per_user`."""
        cap = tokenizer.config.max_events_per_user
        return rows.lo if cap is None else np.maximum(rows.lo, rows.hi - cap)

    def corpus_shape(
        self, rows: CacheRows, tokenizer: KeyValueTimeTokenizer, max_length: int
    ) -> tuple[int, int, int]:
        """`(events, tokens, profile tokens)` one `collate` over all of `rows` pads to."""
        if not len(rows):
            return 1, 1, 1
        lo = self.capped(rows, tokenizer)
        max_events = max(int((rows.hi - lo).max()), 1)
        counts = np.diff(np.asarray(self.token_offsets))
        max_tok = max(
            max(int(counts[a:b].max()) for a, b in zip(lo.tolist(), rows.hi.tolist(), strict=True)),
            1,
        )
        max_prof = max(
            max(
                len(tokenizer.encode_profile(self.user_ids[u], rows.cutoff)[0])
                for u in np.unique(rows.user).tolist()
            ),
            1,
        )
        return max_events, max(min(max_tok, max_length), 1), max_prof

    def collate(
        self,
        rows: CacheRows,
        tokenizer: KeyValueTimeTokenizer,
        *,
        max_length: int,
        pad_to: tuple[int, int, int] | None = None,
    ) -> KeyValueTimeBatch:
        """`collate_batch` for cached rows, without a label window.

        `pad_to` is `corpus_shape(...)` of a larger corpus these rows are cut
        from. Fraud labels are zero: the cache holds no labels, and a run that
        reads one checks the book carries none on its events.
        """
        lo, hi = self.capped(rows, tokenizer), rows.hi
        n_ev = hi - lo
        bsz = len(rows)
        own = self.corpus_shape(rows, tokenizer, max_length)
        max_events, max_tok, max_prof = own if pad_to is None else pad_to

        event_index = _ranges(lo, hi)
        row_of_event = np.repeat(np.arange(bsz, dtype=np.int64), n_ev)
        slot_of_event = np.arange(len(event_index), dtype=np.int64) - np.repeat(
            np.cumsum(n_ev) - n_ev, n_ev
        )
        tok_lo = np.asarray(self.token_offsets[event_index], np.int64)
        tok_n = np.asarray(self.token_offsets[event_index + 1], np.int64) - tok_lo
        take = np.minimum(tok_n, max_tok)
        tok_index = _ranges(tok_lo, tok_lo + take)
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
        key_ids[tok_where] = torch.from_numpy(np.asarray(self.key_ids[tok_index], np.int64))
        value_ids[tok_where] = torch.from_numpy(np.asarray(self.value_ids[tok_index], np.int64))
        positions[tok_where] = torch.from_numpy(np.asarray(self.positions[tok_index], np.int64))
        token_mask[tok_where] = 1

        ev_where = (torch.from_numpy(row_of_event), torch.from_numpy(slot_of_event))
        event_mask = torch.zeros((bsz, max_events), dtype=torch.long)
        event_mask[ev_where] = 1
        id_us = np.asarray(self.id_us[event_index], np.int64)
        time_log = torch.zeros((bsz, max_events), dtype=torch.float32)
        elapsed_log = torch.zeros((bsz, max_events), dtype=torch.float32)
        if len(id_us):
            ends = np.cumsum(n_ev)
            last = np.repeat(id_us[ends - 1], n_ev)
            first = np.repeat(id_us[ends - n_ev], n_ev)
            time_log[ev_where] = torch.tensor(_log_times(last - id_us), dtype=torch.float32)
            elapsed_log[ev_where] = torch.tensor(_log_times(id_us - first), dtype=torch.float32)
        calendar: dict[str, torch.Tensor] = {}
        for name, source in (("hour", self.hour), ("dow", self.dow), ("dom", self.dom)):
            tensor = torch.zeros((bsz, max_events), dtype=torch.long)
            tensor[ev_where] = torch.from_numpy(np.asarray(source[event_index], np.int64))
            calendar[name] = tensor

        entity_ids = tuple(self.user_ids[u] for u in rows.user.tolist())
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
            hour=calendar["hour"],
            dow=calendar["dow"],
            dom=calendar["dom"],
            **_profile_tensors(tokenizer, entity_ids, rows.cutoff, max_prof),
            entity_ids=entity_ids,
            sequence_ids=tuple(f"{e}:{cutoff}" for e in entity_ids),
            fraud_labels=torch.zeros(bsz, dtype=torch.float32),
            numeric_features=self._raw_features(rows.lo, rows.hi),
        )

    def _raw_features(self, lo: NDArray[np.int64], hi: NDArray[np.int64]) -> torch.Tensor:
        """`raw_feature_vector` of each whole row (not capped), summed as Python's `sum` does."""
        out: list[list[float]] = []
        for a, b in zip(lo.tolist(), hi.tolist(), strict=True):
            if b <= a:
                out.append([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            amounts = np.asarray(self.amount[a:b], np.float64).tolist()
            out.append(
                [
                    float(sum(amounts) / len(amounts)),
                    float(max(amounts)),
                    float(b - a),
                    float(self.hour[b - 1]),
                    float(self.dow[b - 1]),
                ]
            )
        return torch.tensor(out, dtype=torch.float32).reshape(len(lo), 5)


def _profile_tensors(
    tokenizer: KeyValueTimeTokenizer, entities: Sequence[str], cutoff: datetime, max_prof: int
) -> dict[str, torch.Tensor]:
    """Profile prefixes, padded as `KeyValueTimeTokenizer.collate` pads them."""
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
    """

    def __init__(
        self,
        cache: TokenCache,
        rows: CacheRows,
        tokenizer: KeyValueTimeTokenizer,
        *,
        batch_size: int,
        max_length: int,
        shuffle_seed: int | None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
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
        chosen = self.order[index * self.batch_size : (index + 1) * self.batch_size]
        return self.cache.collate(
            self.rows.take(chosen), self.tokenizer, max_length=self.max_length, pad_to=self.pad_to
        )


class CachedKeyValueTimeTokenizer(KeyValueTimeTokenizer):
    """The cache's fitted tokenizer; `encode` reads each event's tokens from the cache.

    Events are looked up by `event_id`, so a sequence may hold any cached
    events. An event the cache does not hold, such as a splice's copy, needs an
    alias to the event it was copied from (`add_aliases`).
    """

    cache: TokenCache
    _aliases: dict[str, str]

    @classmethod
    def from_cache(
        cls, cache: TokenCache, profiles: Mapping[str, EntityProfile] | None = None
    ) -> CachedKeyValueTimeTokenizer:
        tokenizer = cls.from_state(cache.tokenizer_state, profiles)
        if not isinstance(tokenizer, CachedKeyValueTimeTokenizer):
            raise TypeError("from_state must build the class it is called on.")
        tokenizer.cache = cache
        tokenizer._aliases = {}
        return tokenizer

    def add_aliases(self, aliases: Mapping[str, str]) -> None:
        """Event ids that stand for other cached events."""
        self._aliases.update(aliases)

    def fit(self, sequences: Sequence[TransactionSequence]) -> KeyValueTimeTokenizer:
        raise UnsupportedCombinationError(
            "A cached tokenizer was fit when its cache was built; it is not refit per run."
        )

    def encode(self, sequence: TransactionSequence) -> EncodedKVTRecord:
        events = list(sequence.events)
        cap = self.config.max_events_per_user
        if cap is not None and len(events) > cap:
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
            hour=tuple(np.asarray(cache.hour[where], np.int64).tolist()),
            dow=tuple(np.asarray(cache.dow[where], np.int64).tolist()),
            dom=tuple(np.asarray(cache.dom[where], np.int64).tolist()),
            prof_key_ids=tuple(prof_k),
            prof_value_ids=tuple(prof_v),
            prof_positions=tuple(prof_p),
            prof_offsets=tuple(prof_off),
            prof_time_log=tuple(prof_t),
        )

    def manifest(self) -> dict[str, object]:
        return {**super().manifest(), "token_cache": str(self.cache.path)}


__all__ = ["CacheRows", "CachedBatches", "CachedKeyValueTimeTokenizer", "TokenCache"]
