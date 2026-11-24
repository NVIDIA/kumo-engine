# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A fitted vocabulary of event field values (`ValueVocabulary`), the part every field
tokenizer shares, and the key-value layout over it (`KeyValuePairs`), each with a
per-sequence and a block-wise (vectorised) path that agree."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Self, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import NDArray

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.schema import EntityProfile, TransactionEvent, TransactionSequence
from tfm.contracts.tokenizers import FIELD_CATEGORICAL, FIELD_NUMERIC, FIELD_TEXT, EventBlock

PAD, MASK, UNK, USR, EVT = 0, 1, 2, 3, 4
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[MASK]", "[UNK]", "[USR]", "[EVT]")


def murmur3_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3, x86 32-bit."""
    c1, c2, mask = 0xCC9E2D51, 0x1B873593, 0xFFFFFFFF
    n, h = len(data), seed
    tail = n - n % 4
    for i in range(0, tail, 4):
        k = int.from_bytes(data[i : i + 4], "little") * c1 & mask
        h ^= ((k << 15) | (k >> 17)) * c2 & mask
        h = (((h << 13) | (h >> 19)) * 5 + 0xE6546B64) & mask
    if tail < n:
        k = int.from_bytes(data[tail:], "little") * c1 & mask
        h ^= ((k << 15) | (k >> 17)) * c2 & mask
    h ^= n
    h = (h ^ h >> 16) * 0x85EBCA6B & mask
    h = (h ^ h >> 13) * 0xC2B2AE35 & mask
    return h ^ h >> 16


def _rotl(x: NDArray[np.uint32], r: int) -> NDArray[np.uint32]:
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def _word(data: NDArray[np.uint8], at: NDArray[np.int64]) -> NDArray[np.uint32]:
    """Little-endian uint32 at each byte offset `at`."""
    w = data[at].astype(np.uint32)
    for i in (1, 2, 3):
        w |= data[at + i].astype(np.uint32) << np.uint32(8 * i)
    return w


def murmur3_32_many(
    offsets: NDArray[np.int64], data: NDArray[np.uint8], seed: int = 0
) -> NDArray[np.uint32]:
    """`murmur3_32` of every byte string `data[offsets[i]:offsets[i + 1]]`."""
    c1, c2 = np.uint32(0xCC9E2D51), np.uint32(0x1B873593)
    start = offsets[:-1].astype(np.int64)
    n = np.diff(offsets).astype(np.int64)
    data = np.concatenate([data, np.zeros(4, np.uint8)])
    h = np.full(len(n), seed, np.uint32)
    blocks = n // 4
    live = np.arange(len(n))
    for j in range(int(blocks.max(initial=0))):
        live = live[blocks[live] > j]
        k = _rotl(_word(data, start[live] + 4 * j) * c1, 15) * c2
        h[live] = _rotl(h[live] ^ k, 13) * np.uint32(5) + np.uint32(0xE6546B64)
    tail = (n % 4).astype(np.uint32)
    live = np.flatnonzero(tail)
    at = start[live] + 4 * blocks[live]
    k = _word(data, at) & (np.uint32(0xFFFFFFFF) >> (np.uint32(8) * (np.uint32(4) - tail[live])))
    h[live] ^= _rotl(k * c1, 15) * c2
    h ^= n.astype(np.uint32)
    h = (h ^ (h >> np.uint32(16))) * np.uint32(0x85EBCA6B)
    h = (h ^ (h >> np.uint32(13))) * np.uint32(0xC2B2AE35)
    return h ^ (h >> np.uint32(16))


def hash_bucket(key: str, value: str, buckets: int) -> int:
    """A process-independent bucket for `(key, value)`; Python's `hash` is salted per run."""
    return murmur3_32(f"{key}\x1f{value}".encode()) % buckets


def _blake2b_bucket(key: str, value: str, buckets: int) -> int:
    """`hash_bucket` of a tokenizer state saved without a `hash`."""
    digest = hashlib.blake2b(f"{key}\x1f{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


_BUCKETS = {"murmur3": hash_bucket, "blake2b": _blake2b_bucket}


def _utf8_buffers(strings: pa.Array) -> tuple[NDArray[np.int64], NDArray[np.uint8]]:
    """`(offsets, bytes)` of a `large_string` array with no nulls."""
    _, offsets, data = strings.buffers()
    first = strings.offset
    offsets_np = np.frombuffer(offsets, np.int64)[first : first + len(strings) + 1]
    data_np = np.frombuffer(data, np.uint8) if data is not None else np.zeros(0, np.uint8)
    return offsets_np, data_np


def event_fields(event: TransactionEvent) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in event.categorical.items():
        fields[str(key)] = str(value)
    for key, value in event.numeric.items():
        fields[str(key)] = str(value)
    for key, value in event.text.items():
        fields[str(key)] = str(value)
    return fields


def _with_source(block: EventBlock) -> EventBlock:
    """Supply the object encoder's implicit source only while encoding."""
    source = block.keys.index("source") if "source" in block.keys else len(block.keys)
    if source < len(block.keys):
        events = block.pair_event[block.pair_key == source]
        if len(events) == block.n_events:
            return block  # Duplicate pairs, if any, are rejected by encode_events.
        present = np.zeros(block.n_events, dtype=bool)
        present[events] = True
        missing = np.flatnonzero(~present)
    else:
        missing = np.arange(block.n_events, dtype=np.int64)
    if not len(missing):
        return block
    value = (
        block.values.index("transaction") if "transaction" in block.values else len(block.values)
    )
    return replace(
        block,
        keys=block.keys if source < len(block.keys) else (*block.keys, "source"),
        values=block.values if value < len(block.values) else (*block.values, "transaction"),
        numbers=block.numbers if value < len(block.values) else np.r_[block.numbers, np.nan],
        pair_event=np.r_[block.pair_event, missing],
        pair_key=np.r_[block.pair_key, np.full(len(missing), source, dtype=np.int64)],
        pair_value=np.r_[block.pair_value, np.full(len(missing), value, dtype=np.int64)],
        pair_kind=np.r_[block.pair_kind, np.full(len(missing), FIELD_CATEGORICAL, dtype=np.int8)],
        pair_map_position=np.r_[block.pair_map_position, np.zeros(len(missing), dtype=np.int64)],
    )


@dataclass
class PercentileBinner:
    n_buckets: int = 16
    edges: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def fit(self, values: np.ndarray, counts: np.ndarray | None = None) -> PercentileBinner:
        """Edges at the percentiles of `values`, each repeated `counts` times if given."""
        nz = np.asarray(values, dtype=np.float64)
        keep = np.isfinite(nz) & (nz != 0.0)
        nz = nz[keep]
        if len(nz) == 0:
            self.edges = np.zeros(0, dtype=np.float64)
            return self
        qs = np.linspace(0, 1, self.n_buckets + 1)[1:-1]
        if counts is None:
            self.edges = np.unique(np.quantile(nz, qs))
        else:
            self.edges = np.unique(_repeated_quantile(nz, np.asarray(counts)[keep], qs))
        return self

    @property
    def n_bins(self) -> int:
        return int(len(self.edges) + 2)

    def transform_one(self, value: float) -> int:
        x = float(value)
        if not math.isfinite(x) or x == 0.0:
            return 0
        return int(np.searchsorted(self.edges, x, side="right") + 1)


def _repeated_quantile(values: np.ndarray, counts: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """`np.quantile(np.repeat(values, counts), qs)`, bit for bit, without repeating."""
    order = np.argsort(values, kind="stable")
    values, ends = values[order], np.cumsum(counts[order], dtype=np.int64)
    n = int(ends[-1])
    # numpy's "linear" method: virtual index, floor neighbours, then `_lerp`.
    virtual = (n - 1) * qs
    lo = np.minimum(np.floor(virtual), n - 1).astype(np.int64)
    a = values[np.searchsorted(ends, lo, side="right")]
    b = values[np.searchsorted(ends, np.minimum(lo + 1, n - 1), side="right")]
    t = virtual - lo
    diff = b - a
    return np.where(t >= 0.5, b - diff * (1 - t), a + diff * t)


@dataclass
class KeyFitStatistics:
    """What `fit` saw for one key: value counts and numeric readings."""

    counts: dict[str, int]
    """Sample count per value, in the order each value first appeared. Holds every
    value, or only `VocabularyConfig.fit_value_cap(key)` of them when there are more."""
    n_samples: int
    numeric: list[float] | np.ndarray
    """Numeric readings: each one, or each distinct one once with `numeric_counts`."""
    numeric_counts: np.ndarray | None = None
    """How often each `numeric` reading occurred; `None` counts each once."""

    @property
    def n_numeric(self) -> int:
        """Number of numeric readings."""
        return len(self.numeric) if self.numeric_counts is None else int(self.numeric_counts.sum())


@dataclass
class FitStatistics:
    """Everything `fit` needs, collected from sequences or built column-wise.

    `keys` is in the order each key first appeared: vocabulary ids and the
    tie-break among equally frequent values both follow it.
    """

    keys: dict[str, KeyFitStatistics]


_BLOCK_SHIFT = 1 << 42
"""Fit ordinals are `block * _BLOCK_SHIFT + local`, so blocks keep entity order."""
_PROFILE_GROUP = 3


@dataclass
class KeyBlockStatistics:
    """One key's statistics from some blocks: what `merge` combines."""

    min_ord: int
    """Ordinal of the key's first sample: its place in first-appearance order."""
    n: int
    values: dict[str, list[int]]
    """Each kept value's `[count, ordinal of first sample]`."""
    numeric: tuple[NDArray[np.float64], NDArray[np.int64]]
    """Distinct numeric readings, ascending, with their counts."""
    pending: list[tuple[NDArray[np.float64], NDArray[np.int64]]]
    """Readings not yet folded into `numeric`."""


@dataclass
class KeyValueStats:
    """`ValueVocabulary.block_statistics`, combined with `ValueVocabulary.merge`."""

    keys: dict[str, KeyBlockStatistics]

    def fit_statistics(self) -> FitStatistics:
        """What `ValueVocabulary.fit` reads: keys and values in the order they
        first appeared."""
        keys: dict[str, KeyFitStatistics] = {}
        for key in sorted(self.keys, key=lambda k: self.keys[k].min_ord):
            entry = self.keys[key]
            ordered = sorted(entry.values.items(), key=lambda kv: kv[1][1])
            numeric, numeric_counts = _fold([entry.numeric, *entry.pending])
            keys[key] = KeyFitStatistics(
                counts={value: count for value, (count, _) in ordered},
                n_samples=entry.n,
                numeric=numeric,
                numeric_counts=numeric_counts,
            )
        return FitStatistics(keys=keys)


def _runs(sorted_: NDArray[np.int64]) -> NDArray[np.int64]:
    """Start of each run of equal values in `sorted_`, then `len(sorted_)`."""
    if not len(sorted_):
        return np.zeros(1, np.int64)
    return np.r_[np.flatnonzero(np.r_[True, sorted_[1:] != sorted_[:-1]]), len(sorted_)]


def _put(values: dict[str, list[int]], value: str, count: int, ord_: int) -> None:
    """Add `count` samples of `value`, first seen at `ord_`."""
    slot = values.get(value)
    if slot is None:
        values[value] = [count, ord_]
    else:
        slot[0] += count
        slot[1] = min(slot[1], ord_)


def _trim(values: dict[str, list[int]], cap: int | None) -> None:
    """Keep the `cap` smallest values by string: the same set for any merge order."""
    if cap is not None and len(values) > cap:
        for value in sorted(values)[cap:]:
            del values[value]


_NO_READINGS = (np.zeros(0, np.float64), np.zeros(0, np.int64))


def _fold(
    readings: list[tuple[NDArray[np.float64], NDArray[np.int64]]],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Distinct readings, ascending, with summed counts; `-0.0` counts as `0.0`."""
    values = np.concatenate([v for v, _ in readings]) + 0.0
    uniq, inverse = np.unique(values, return_inverse=True)
    counts = np.bincount(inverse, weights=np.concatenate([c for _, c in readings]))
    return uniq, counts.astype(np.int64)


def _unique_inverse(
    code: NDArray[np.int64], size: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """`np.unique(code, return_inverse=True)` for codes in `[0, size)`."""
    if size > len(code):
        uniq, inverse = np.unique(code, return_inverse=True)
        return uniq, inverse.reshape(-1)
    seen = np.zeros(size, bool)
    seen[code] = True
    return np.flatnonzero(seen), (np.cumsum(seen, dtype=np.int64) - 1)[code]


@dataclass(frozen=True)
class VocabularyConfig:
    """How a `ValueVocabulary` sorts keys into kinds and sizes their values."""

    n_buckets: int = 16
    """Quantile bins per numeric key."""
    categorical_threshold: int = 64
    """Most values a key may have and stay categorical; the most frequent are kept."""
    numeric_min_cardinality: int = 8
    """Fewest distinct values a key needs to be numeric."""
    categorical_keys: tuple[str, ...] = ()
    """Keys kept categorical even when their values parse as numbers (codes such as `mcc`)."""
    hashed_keys: tuple[str, ...] = ()
    """High-cardinality identity keys (merchant, device, counterparty) encoded as one
    token, a stable hash bucket of the value, instead of its bytes."""
    hash_buckets: int = 8192
    """Buckets per hashed key, each key with its own range: the output layer is
    tied to the input embedding, so a bucket shared across keys would have to
    be a merchant, a device, and a counterparty at once. Nothing is fitted: an
    unseen value gets a bucket too, and two values of one key that collide
    share it."""
    excluded_keys: tuple[str, ...] = ()
    """Event keys never fitted or encoded: identity keys such as TabFormer's `user`,
    which repeats one value in every event of a history."""

    def fit_value_cap(self, key: str) -> int | None:
        """Distinct values of `key` that `fit` needs; `None` means all.

        Past the cap, `fit` compares only the number of distinct values, and the cap
        exceeds both thresholds it compares that number with.
        """
        if key in self.categorical_keys:
            return None
        return max(self.categorical_threshold, self.numeric_min_cardinality) + 1


class ValueVocabulary:
    """Token ids for event field values (experimental until the packed batch, P2.7).

    Each key gets a kind and its own id range: a quantile bin per numeric value
    (`VAL_<key>_b<bucket>`), a token per categorical value (`VAL_<key>_<value>`), a hash
    bucket per hashed value (`VAL_<key>_h<bucket>`); text is up to 32 UTF-8 byte tokens
    shared by all keys. A value id thus names its key, except for text.

    Fit it from sequences (`collect`, then `fit`) or block by block (`block_statistics`,
    `merge`, then `fit`): both give the same vocabulary. A layout with tokens of its own
    before the value ids calls `fit_kinds`, `add_token`, then `fit_values`. Encode one
    value at a time (`encode_value`) or a whole block at once (`value_ids`,
    `encode_events`): both give the same ids.
    """

    def __init__(
        self, config: VocabularyConfig | None = None, special_tokens: Sequence[str] = SPECIAL_TOKENS
    ) -> None:
        self.config = config or VocabularyConfig()
        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(special_tokens)}
        self.id_to_token: dict[int, str] = dict(enumerate(special_tokens))
        self.field_kind: dict[str, str] = {}
        self.binners: dict[str, PercentileBinner] = {}
        self.binner_base: dict[str, int] = {}
        self.cat_vocab: dict[str, dict[str, int]] = {}
        self.hash_base: dict[str, int] = {}
        self.bucket_hash = "murmur3"
        self.byte_offset: int = 0
        self.fitted = False

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def add_token(self, token: str) -> int:
        """`token`'s id, added at the end if new."""
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        return self.token_to_id[token]

    # Fit -------------------------------------------------------------------

    def collect(
        self, sequences: Sequence[TransactionSequence], profiles: Mapping[str, EntityProfile]
    ) -> FitStatistics:
        """What `fit` reads from `sequences` and their entities' `profiles`, in that order."""
        keys: dict[str, KeyFitStatistics] = {}

        def sample(key: str, raw: str) -> KeyFitStatistics:
            stats = keys.get(key)
            if stats is None:
                stats = keys[key] = KeyFitStatistics(counts={}, n_samples=0, numeric=[])
            stats.counts[raw] = stats.counts.get(raw, 0) + 1
            stats.n_samples += 1
            return stats

        excluded = set(self.config.excluded_keys)
        for seq in sequences:
            for event in seq.events:
                for key, raw in event_fields(event).items():
                    if key in excluded:
                        continue
                    stats = sample(key, raw)
                    numeric = cast(list[float], stats.numeric)
                    if key in event.numeric:
                        numeric.append(float(event.numeric[key]))
                    else:
                        try:
                            parsed = float(raw)
                        except ValueError:
                            continue
                        if math.isfinite(parsed):
                            numeric.append(parsed)
            profile = profiles.get(seq.entity_id)
            if profile is not None:
                for key, raw in profile.attributes.items():
                    sample(key, raw)
                for item in profile.lifelong:
                    if item.timestamp <= seq.cutoff_time:
                        sample(item.key, item.key)
        return FitStatistics(keys=keys)

    def fit(self, statistics: FitStatistics) -> Self:
        """`fit_kinds`, then `fit_values`."""
        return self.fit_kinds(statistics).fit_values(statistics)

    def fit_kinds(self, statistics: FitStatistics) -> Self:
        """Each key's kind: hashed, categorical, numeric or text."""
        if self.fitted:
            raise UnsupportedCombinationError(
                "This vocabulary is already fitted; fit a new tokenizer instead of refitting."
            )
        config = self.config
        for key, stats in statistics.keys.items():
            distinct = len(stats.counts)
            n_numeric = stats.n_numeric
            parse_ratio = n_numeric / max(stats.n_samples, 1)
            if key in config.hashed_keys:
                self.field_kind[key] = "hashed"
            elif key in config.categorical_keys:
                self.field_kind[key] = "categorical"
            elif (
                parse_ratio >= 0.8 and n_numeric >= 4 and distinct >= config.numeric_min_cardinality
            ):
                self.field_kind[key] = "numeric"
            elif distinct <= config.categorical_threshold:
                self.field_kind[key] = "categorical"
            else:
                self.field_kind[key] = "text"
        return self

    def fit_values(self, statistics: FitStatistics) -> Self:
        """Value tokens after the existing ids: bins and categories, hash ranges, bytes."""
        config = self.config

        def new(token: str) -> int:
            if token in self.token_to_id:
                raise SchemaError(
                    f"Two fields map to the value token {token!r} (e.g. a='b_c' and "
                    "a_b='c'); rename one of the keys."
                )
            return self.add_token(token)

        for key, kind in self.field_kind.items():
            if kind == "numeric":
                binner = PercentileBinner(config.n_buckets)
                stats = statistics.keys[key]
                binner.fit(np.asarray(stats.numeric, dtype=np.float64), stats.numeric_counts)
                self.binners[key] = binner
                self.binner_base[key] = len(self.token_to_id)
                for bucket in range(binner.n_bins):
                    new(f"VAL_{key}_b{bucket}")
            elif kind == "categorical":
                counts = Counter(statistics.keys[key].counts)
                self.cat_vocab[key] = {}
                for value, _ in counts.most_common(config.categorical_threshold):
                    self.cat_vocab[key][value] = new(f"VAL_{key}_{value}")

        for key in sorted(k for k, kind in self.field_kind.items() if kind == "hashed"):
            self.hash_base[key] = len(self.token_to_id)
            for bucket in range(config.hash_buckets):
                new(f"VAL_{key}_h{bucket}")
        self.byte_offset = len(self.token_to_id)
        for byte in range(256):
            self.add_token(f"BYTE_{byte}")
        self.fitted = True
        return self

    def block_statistics(
        self, block: EventBlock, profiles: Mapping[str, EntityProfile]
    ) -> KeyValueStats:
        """`collect` of `block`'s events and `profiles`: per-key sample counts and
        first-appearance ordinals, value counts up to `fit_value_cap`, and each distinct
        numeric reading with its count."""
        config = self.config
        excluded = np.isin(np.array(block.keys, object), list(config.excluded_keys))
        mask = ~excluded[block.pair_key]
        event = block.pair_event[mask]
        key = block.pair_key[mask]
        value = block.pair_value[mask]
        kind = block.pair_kind[mask]
        group = np.where(kind == FIELD_NUMERIC, 1, np.where(kind == FIELD_TEXT, 2, 0))
        # Slots wide enough for the block's widest event, so ordinals sort as `collect` reads.
        width = max(128, int(block.pair_map_position.max(initial=0)) + 1)
        if (block.n_events + 1) * 4 * width > _BLOCK_SHIFT:
            raise SchemaError(
                f"Block {block.index} has {block.n_events} events, the widest with {width} "
                "fields: too many to order within a block. Lower rows_per_block, or send "
                "fewer fields per event."
            )
        ordinal = (event * 4 + group) * width + block.pair_map_position[mask]
        base = block.index * _BLOCK_SHIFT

        n_values = len(block.values)
        code = key * n_values + value
        order = np.lexsort((ordinal, code))
        code_sorted = code[order]
        first = _runs(code_sorted)[:-1]
        counts = np.diff(np.r_[first, len(code)])
        first_ord = base + ordinal[order[first]]
        pair_key, pair_value = np.divmod(code_sorted[first], n_values)
        keys: dict[str, KeyBlockStatistics] = {}
        # Codes sort by key, so each key's distinct values are one run.
        for a, b in itertools.pairwise(_runs(pair_key).tolist()):
            name = block.keys[int(pair_key[a])]
            cap = config.fit_value_cap(name)
            kept = slice(a, b if cap is None else min(b, a + cap))
            keys[name] = KeyBlockStatistics(
                min_ord=int(first_ord[a:b].min()),
                n=int(counts[a:b].sum()),
                values={
                    block.values[v]: [c, o]
                    for v, c, o in zip(
                        pair_value[kept].tolist(),
                        counts[kept].tolist(),
                        first_ord[kept].tolist(),
                        strict=True,
                    )
                },
                numeric=_NO_READINGS,
                pending=[],
            )
        # Every pair whose value parses as a finite float is a reading, as in `collect`.
        reading = (kind == FIELD_NUMERIC) | np.isfinite(block.numbers[value])
        num_code, num_count = np.unique(code[reading], return_counts=True)
        num_key, num_value = np.divmod(num_code, n_values)
        for a, b in itertools.pairwise(_runs(num_key).tolist()):
            keys[block.keys[int(num_key[a])]].pending.append(
                (block.numbers[num_value[a:b]].astype(np.float64), num_count[a:b])
            )
        ends = block.user_offsets[1:].tolist()
        for u, user in enumerate(block.user_ids):
            profile = profiles.get(user)
            if profile is None or ends[u] == block.user_offsets[u]:
                continue
            items = [(k, v) for k, v in profile.attributes.items()]
            items += [(i.key, i.key) for i in profile.lifelong]
            if len(items) > 128:
                raise SchemaError(f"Profile {profile.entity_id!r} has more than 128 fit items.")
            anchor = base + ((ends[u] - 1) * 4 + _PROFILE_GROUP) * width
            for index, (name, raw) in enumerate(items):
                entry = keys.setdefault(
                    name, KeyBlockStatistics(anchor + index, 0, {}, _NO_READINGS, [])
                )
                entry.n += 1
                entry.min_ord = min(entry.min_ord, anchor + index)
                _put(entry.values, raw, 1, anchor + index)
        for name, entry in keys.items():
            _trim(entry.values, config.fit_value_cap(name))
        return KeyValueStats(keys)

    def merge(self, a: KeyValueStats, b: KeyValueStats) -> KeyValueStats:
        """`a` and `b` combined, the same in any order; reuses `a`'s storage.

        Memory grows with each key's kept values (every value of a declared categorical
        key, else at most `fit_value_cap`) and distinct numeric readings, not with the
        number of blocks: a key's readings are folded once its unfolded ones outnumber
        its folded ones.
        """
        for key, entry in b.keys.items():
            target = a.keys.get(key)
            if target is None:
                target = a.keys[key] = KeyBlockStatistics(entry.min_ord, 0, {}, _NO_READINGS, [])
            target.n += entry.n
            target.min_ord = min(target.min_ord, entry.min_ord)
            for value, (count, ord_) in entry.values.items():
                _put(target.values, value, count, ord_)
            _trim(target.values, self.config.fit_value_cap(key))
            if len(entry.numeric[0]):
                target.pending.append(entry.numeric)
            target.pending.extend(entry.pending)
            if sum(len(v) for v, _ in target.pending) > len(target.numeric[0]):
                target.numeric = _fold([target.numeric, *target.pending])
                target.pending = []
        return a

    # Encode, one value at a time ---------------------------------------------

    def encode_value(self, key: str, value: str, value_ids: list[int], positions: list[int]) -> int:
        """Append one field's value tokens and their byte positions, and return how many:
        one token, or up to 32 bytes for text. An unfitted key, or a numeric value that
        is not a finite number, appends `[UNK]`."""
        kind = self.field_kind.get(key)
        if kind is None:
            value_ids.append(UNK)
            positions.append(0)
        elif kind == "numeric":
            try:
                parsed = float(value)
            except ValueError:
                parsed = float("nan")
            if math.isfinite(parsed):
                value_ids.append(self.binner_base[key] + self.binners[key].transform_one(parsed))
            else:
                value_ids.append(UNK)
            positions.append(0)
        elif kind == "categorical":
            value_ids.append(self.cat_vocab[key].get(value, UNK))
            positions.append(0)
        elif kind == "hashed":
            value_ids.append(
                self.hash_base[key]
                + _BUCKETS[self.bucket_hash](key, value, self.config.hash_buckets)
            )
            positions.append(0)
        else:
            data = value.encode("utf-8")[:32] or b"\x00"
            offset = self.byte_offset
            value_ids.extend([offset + byte for byte in data])
            positions.extend(range(len(data)))
            return len(data)
        return 1

    def value_tokens(self, key: str, value: str) -> tuple[list[int], list[int]]:
        """`(value ids, positions)` for one field, before any per-event cap."""
        value_ids: list[int] = []
        positions: list[int] = []
        self.encode_value(key, value, value_ids, positions)
        return value_ids, positions

    def ordered_fields(self, event: TransactionEvent) -> list[tuple[str, str]]:
        """`event`'s fields without excluded keys, in `field_order`; `source` defaults to
        `"transaction"`."""
        fields = event_fields(event)
        for key in self.config.excluded_keys:
            fields.pop(key, None)
        source = fields.pop("source", "transaction")
        kind = self.field_kind
        rest = sorted(fields.items(), key=lambda item: (kind.get(item[0]) == "text", item[0]))
        return [("source", source), *rest]

    def field_order(self, key: str) -> tuple[int, bool, str]:
        """Where a field sits in its event: `source`, single-token fields by key, then text,
        so a per-event cap only ever trims text."""
        return (0 if key == "source" else 1, self.field_kind.get(key) == "text", key)

    # Encode, a block at once -------------------------------------------------

    def value_ids(
        self, key: str, values: pa.Array, numbers: NDArray[np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """`value_tokens(key, v)` for every `v` in `values`, flattened: `(lengths, ids,
        positions)`, int64. `numbers[i]` is `values[i]` as a finite float, else NaN."""
        n = len(values)
        one = np.ones(n, np.int64)
        zeros = np.zeros(n, np.int64)
        kind = self.field_kind.get(key)
        if kind is None:
            return one, np.full(n, UNK, np.int64), zeros
        if kind == "numeric":
            edges = self.binners[key].edges
            bucket = np.searchsorted(edges, numbers, side="right") + 1
            bucket[numbers == 0.0] = 0
            ids = np.where(np.isfinite(numbers), self.binner_base[key] + bucket, UNK)
            return one, ids.astype(np.int64), zeros
        if kind == "categorical":
            vocab = self.cat_vocab[key]
            found = pc.index_in(values, value_set=pa.array(list(vocab), values.type))  # type: ignore[attr-defined]
            ids = np.asarray(list(vocab.values()) + [UNK], np.int64)
            return one, ids[found.fill_null(len(vocab)).to_numpy()], zeros
        if kind == "hashed":
            if self.bucket_hash != "murmur3":
                raise UnsupportedCombinationError(
                    f"This tokenizer state hashes with {self.bucket_hash}; batch encoding "
                    "supports murmur3 only. Refit the tokenizer, or rebuild the token cache."
                )
            sep = pa.scalar("\x1f", values.type)
            joined = pc.binary_join_element_wise(pa.scalar(key, values.type), values, sep)  # type: ignore[attr-defined]
            hashes = murmur3_32_many(*_utf8_buffers(joined))
            buckets = (hashes % self.config.hash_buckets).astype(np.int64)
            return one, self.hash_base[key] + buckets, zeros
        offsets, data = _utf8_buffers(values)
        lengths = np.minimum(np.diff(offsets), 32)
        empty = lengths == 0
        counts = np.maximum(lengths, 1)
        positions = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
        data = np.concatenate([data, np.zeros(1, np.uint8)])
        at = np.repeat(np.where(empty, len(data) - 1, offsets[:-1]), counts) + positions
        return counts, self.byte_offset + data[at].astype(np.int64), positions

    def encode_events(
        self,
        block: EventBlock,
        max_event_tokens: int | None,
        key_labels: NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], int]:
        """Every event's value tokens in `ordered_fields` order, each event cut to
        `max_event_tokens`: `(token_offsets, token_keys, value_ids, positions, events cut)`,
        int64. `token_keys[t]` is `key_labels[k]` for the key `block.keys[k]` of token `t`
        (default: `k`). Excluded keys get no tokens; an excluded `source` still opens its
        event, as `[UNK]`."""
        old_keys = len(block.keys)
        block = _with_source(block)
        if key_labels is not None and len(block.keys) != old_keys:
            key_labels = np.r_[key_labels, UNK]
        n_values = len(block.values)
        labels = np.arange(len(block.keys)) if key_labels is None else key_labels
        dropped = list(set(self.config.excluded_keys) - {"source"})
        kept = ~np.isin(np.array(block.keys, object), dropped)[block.pair_key]
        pair_event, pair_key = block.pair_event[kept], block.pair_key[kept]
        code = pair_key * n_values + block.pair_value[kept]
        uniq, inverse = _unique_inverse(code, len(block.keys) * n_values)
        uniq_key, uniq_value = np.divmod(uniq, n_values)
        values = pa.array(block.values, pa.large_string())
        bounds = np.searchsorted(uniq_key, np.arange(len(block.keys) + 1))
        encoded = []
        for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            if b > a:
                lengths, ids, pos = self.value_ids(
                    block.keys[k], values.take(uniq_value[a:b]), block.numbers[uniq_value[a:b]]
                )
                encoded.append((lengths, np.full(len(ids), labels[k], np.int64), ids, pos))
        lengths, flat_k, flat_v, flat_p = (
            np.concatenate([e[i] for e in encoded]) if encoded else np.zeros(0, np.int64)
            for i in range(4)
        )
        uniq_start = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        key_rank = np.empty(len(block.keys), np.int64)
        key_rank[sorted(range(len(block.keys)), key=lambda i: self.field_order(block.keys[i]))] = (
            np.arange(len(block.keys))
        )

        slot = pair_event * (len(block.keys) + 1) + key_rank[pair_key]
        order = np.argsort(slot, kind="stable")
        ordered = slot[order]
        if (ordered[1:] == ordered[:-1]).any():
            raise SchemaError(
                f"Block {block.index} gives an event the same key twice; an EventBlock holds "
                "one pair per (event, key), as event_fields keeps one value per key."
            )
        pair_uniq = inverse[order]
        counts = lengths[pair_uniq]
        total = int(counts.sum())
        before = np.cumsum(counts) - counts
        gather = np.repeat(uniq_start[pair_uniq] - before, counts) + np.arange(total)
        token_event = np.repeat(pair_event[order], counts)
        event_counts = np.bincount(token_event, minlength=block.n_events).astype(np.int64)
        event_first = np.cumsum(event_counts) - event_counts
        rank = np.arange(total) - event_first[token_event]
        cap = max_event_tokens
        keep = np.ones(total, bool) if cap is None else rank < cap
        kept_counts = event_counts if cap is None else np.minimum(event_counts, cap)
        return (
            np.concatenate([[0], np.cumsum(kept_counts)]).astype(np.int64),
            np.asarray(flat_k, np.int64)[gather[keep]],
            np.asarray(flat_v, np.int64)[gather[keep]],
            np.asarray(flat_p, np.int64)[gather[keep]],
            int((event_counts > kept_counts).sum()),
        )

    # State -------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """The fitted vocabulary, JSON-serializable. Hash ranges appear only with hashed
        keys, so a state saved before hashing existed round-trips unchanged."""
        extra: dict[str, Any] = {}
        if self.config.hashed_keys:
            extra["hash_base"] = dict(self.hash_base)
            extra["hash"] = self.bucket_hash
        return {
            **extra,
            "tokens": [self.id_to_token[i] for i in range(len(self.id_to_token))],
            "field_kind": dict(self.field_kind),
            "binner_edges": {k: b.edges.tolist() for k, b in self.binners.items()},
            "binner_n_buckets": {k: b.n_buckets for k, b in self.binners.items()},
            "binner_base": dict(self.binner_base),
            "cat_vocab": {k: dict(v) for k, v in self.cat_vocab.items()},
            "byte_offset": self.byte_offset,
        }

    @classmethod
    def from_state(cls, config: VocabularyConfig, state: Mapping[str, Any]) -> Self:
        """A fitted vocabulary from `state_dict()` and the config it was fitted with."""
        vocabulary = cls(config)
        hash_base = state.get("hash_base", {})
        if not isinstance(hash_base, Mapping):
            raise ValueError(
                "This tokenizer state shares one hash-bucket range across keys; "
                "hashed keys now have a range each. Refit, or rebuild the token cache."
            )
        vocabulary.hash_base = {str(k): int(v) for k, v in hash_base.items()}
        vocabulary.bucket_hash = str(state.get("hash", "blake2b"))
        if vocabulary.bucket_hash not in _BUCKETS:
            raise SchemaError(
                f"Unknown hash {vocabulary.bucket_hash!r} in this tokenizer state. "
                "Refit the tokenizer, or rebuild the token cache."
            )
        vocabulary.token_to_id = {tok: i for i, tok in enumerate(state["tokens"])}
        vocabulary.id_to_token = dict(enumerate(state["tokens"]))
        vocabulary.field_kind = dict(state["field_kind"])
        vocabulary.binners = {
            k: PercentileBinner(
                int(state["binner_n_buckets"][k]), np.asarray(edges, dtype=np.float64)
            )
            for k, edges in state["binner_edges"].items()
        }
        vocabulary.binner_base = {k: int(v) for k, v in state["binner_base"].items()}
        vocabulary.cat_vocab = {k: dict(v) for k, v in state["cat_vocab"].items()}
        vocabulary.byte_offset = int(state["byte_offset"])
        vocabulary.fitted = True
        return vocabulary


class KeyValuePairs:
    """The key-value layout over a `ValueVocabulary` (experimental until the packed
    batch, P2.7): every value token has beside it its key's `KEY_<key>` token, and an
    unfitted key is `([UNK], [UNK])`."""

    def __init__(self, vocabulary: ValueVocabulary) -> None:
        self.vocabulary = vocabulary
        self.key_vocab: dict[str, int] = {}

    def fit(self, statistics: FitStatistics) -> None:
        """Kinds, then one `KEY_` token per key in name order, then the value tokens."""
        vocabulary = self.vocabulary.fit_kinds(statistics)
        for key in sorted(vocabulary.field_kind):
            self.key_vocab[key] = vocabulary.add_token(f"KEY_{key}")
        vocabulary.fit_values(statistics)

    def encode_field(
        self,
        key: str,
        value: str,
        key_ids: list[int],
        value_ids: list[int],
        positions: list[int],
    ) -> None:
        """Append one field's `(key, value, position)` tokens."""
        n = self.vocabulary.encode_value(key, value, value_ids, positions)
        kid = self.key_vocab.get(key, UNK)
        if n == 1:
            key_ids.append(kid)
        else:
            key_ids.extend([kid] * n)

    def encode_event(
        self,
        event: TransactionEvent,
        key_ids: list[int],
        value_ids: list[int],
        positions: list[int],
    ) -> None:
        """Append one event's tokens in `ValueVocabulary.ordered_fields` order."""
        encode_value = self.vocabulary.encode_value
        key_vocab = self.key_vocab
        for key, value in self.vocabulary.ordered_fields(event):
            n = encode_value(key, value, value_ids, positions)
            kid = key_vocab.get(key, UNK)
            if n == 1:
                key_ids.append(kid)
            else:
                key_ids.extend([kid] * n)

    def field_tokens(self, key: str, value: str) -> tuple[list[int], list[int], list[int]]:
        """`(keys, values, positions)` for one field, before any per-event cap."""
        key_ids: list[int] = []
        value_ids: list[int] = []
        positions: list[int] = []
        self.encode_field(key, value, key_ids, value_ids, positions)
        return key_ids, value_ids, positions

    def encode_block(
        self, block: EventBlock, max_event_tokens: int | None
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], int]:
        """`encode_event` of every event in `block`, each cut to `max_event_tokens`:
        `(token_offsets, key_ids, value_ids, positions, events cut)`, int64."""
        block = _with_source(block)
        labels = np.asarray([self.key_vocab.get(k, UNK) for k in block.keys], np.int64)
        return self.vocabulary.encode_events(block, max_event_tokens, labels)

    def field_names(self) -> dict[int, str]:
        """Each `KEY_` id with its key (`FieldNamedTokenizer.field_names`)."""
        return {key_id: key for key, key_id in self.key_vocab.items()}

    def state_dict(self) -> dict[str, Any]:
        """The vocabulary's state with `key_vocab`, JSON-serializable."""
        state = self.vocabulary.state_dict()
        byte_offset = state.pop("byte_offset")
        return {**state, "key_vocab": dict(self.key_vocab), "byte_offset": byte_offset}

    @classmethod
    def from_state(cls, config: VocabularyConfig, state: Mapping[str, Any]) -> Self:
        """Fitted pairs from `state_dict()` and the config the vocabulary was fitted with."""
        pairs = cls(ValueVocabulary.from_state(config, state))
        pairs.key_vocab = dict(state["key_vocab"])
        return pairs


__all__ = [
    "EVT",
    "MASK",
    "PAD",
    "SPECIAL_TOKENS",
    "UNK",
    "USR",
    "FitStatistics",
    "KeyBlockStatistics",
    "KeyFitStatistics",
    "KeyValuePairs",
    "KeyValueStats",
    "PercentileBinner",
    "ValueVocabulary",
    "VocabularyConfig",
    "event_fields",
    "hash_bucket",
    "murmur3_32",
    "murmur3_32_many",
]
