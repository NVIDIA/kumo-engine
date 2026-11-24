# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a key-value-time token cache from a pragmatiq book (ADR 0016).

Two passes over blocks of whole users, each block in its own process:

1. **Fit statistics.** For every user, the train-side sequence `prepare_data`
   would fit on (events in the observation window up to the train cutoff,
   the most recent `max_events`), counted exactly as
   `KeyValueTimeTokenizer.collect_statistics` counts it, including the order
   values first appear in. Profiles are added in the parent process. The
   tokenizer is then fit from these statistics alone.
2. **Encode.** Every event of every user, at every time, becomes the tokens
   `KeyValueTimeTokenizer.encode` would give it. An event's tokens depend
   only on its own fields, so any later sequence is a slice of the cache.

No label is read. Events after the train cutoff are encoded but never counted.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Collection, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.schema import EntityProfile, ensure_utc
from tfm.data.adapters.pragmatiq import load_profiles
from tfm.data.adapters.pragmatiq_columns import (
    FIELD_NUMERIC,
    FIELD_TEXT,
    book_fingerprint,
    plan_blocks,
    read_block,
)
from tfm.features.tokenizers.key_value_time import (
    FitStatistics,
    KeyFitStatistics,
    KeyValueTimeConfig,
    KeyValueTimeTokenizer,
)

FORMAT_VERSION = 1
_BLOCK_SHIFT = 1 << 42
"""Ordinals are `block * _BLOCK_SHIFT + local`, so blocks keep entity order."""
_PROFILE_GROUP = 3


@dataclass(frozen=True)
class FitSpec:
    """Which events the vocabulary is fit on: `prepare_data`'s train-side sequences.

    Train events are those strictly before `train_end` (`assign_temporal_split`);
    each user's sequence keeps those inside the observation window before it,
    then the most recent `max_events`.
    """

    train_end: datetime
    observation_window: timedelta | None
    max_events: int
    """Most recent events per sequence, as `data.sequence_length`; 0 keeps all."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_end": ensure_utc(self.train_end).isoformat(),
            "observation_window": None
            if self.observation_window is None
            else str(self.observation_window),
            "max_events": self.max_events,
        }


def epoch_us(ts: datetime) -> int:
    """Exact microseconds since the epoch."""
    return (ensure_utc(ts) - _EPOCH) // timedelta(microseconds=1)


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def train_ranges(
    ts_us: NDArray[np.int64],
    user_offsets: NDArray[np.int64],
    *,
    train_end: datetime,
    lookback: timedelta | None,
    max_events: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Per user, `[lo, hi)` of events with `train_end - lookback <= ts < train_end`,
    then the most recent `max_events` of them (0 keeps all)."""
    end = epoch_us(train_end)
    start = None if lookback is None else end - lookback // timedelta(microseconds=1)
    lo = np.empty(len(user_offsets) - 1, np.int64)
    hi = np.empty_like(lo)
    for u in range(len(lo)):
        a, b = int(user_offsets[u]), int(user_offsets[u + 1])
        user_ts = ts_us[a:b]
        h = a + int(np.searchsorted(user_ts, end, side="left"))
        low = a if start is None else a + int(np.searchsorted(user_ts, start, side="left"))
        low = min(low, h)
        if max_events > 0:
            low = max(low, h - max_events)
        lo[u], hi[u] = low, h
    return lo, hi


def fit_statistics_block(
    root: str, start: int, stop: int, block: int, fit: FitSpec, users: Collection[str] | None
) -> dict[str, Any]:
    """Pass 1 for one block: per-key counts, first-occurrence ordinals, numeric readings."""
    data = read_block(Path(root), start, stop, users=users)
    lo, hi = train_ranges(
        data.ts_us,
        data.user_offsets,
        train_end=fit.train_end,
        lookback=fit.observation_window,
        max_events=fit.max_events,
    )
    in_fit = np.zeros(data.n_events, bool)
    for a, b in zip(lo.tolist(), hi.tolist(), strict=True):
        in_fit[a:b] = True
    mask = in_fit[data.pair_event]
    event = data.pair_event[mask]
    key = data.pair_key[mask]
    value = data.pair_value[mask]
    kind = data.pair_kind[mask]
    group = np.where(kind == FIELD_NUMERIC, 1, np.where(kind == FIELD_TEXT, 2, 0))
    ordinal = event * 512 + group * 128 + data.pair_map_position[mask]
    base = block * _BLOCK_SHIFT

    n_values = len(data.values)
    code = key * n_values + value
    order = np.lexsort((ordinal, code))
    code_sorted = code[order]
    first = np.flatnonzero(np.r_[True, code_sorted[1:] != code_sorted[:-1]]) if len(code) else []
    counts = np.diff(np.r_[first, len(code_sorted)]) if len(code) else []
    keys: dict[str, dict[str, Any]] = {}
    for idx, count in zip(np.asarray(first).tolist(), np.asarray(counts).tolist(), strict=True):
        pair = int(code_sorted[idx])
        k, v = divmod(pair, n_values)
        entry = keys.setdefault(
            data.keys[k], {"min_ord": None, "n": 0, "values": {}, "numeric": []}
        )
        ord_ = base + int(ordinal[order[idx]])
        entry["n"] += int(count)
        if entry["min_ord"] is None or ord_ < entry["min_ord"]:
            entry["min_ord"] = ord_
        slot = entry["values"].get(data.values[v])
        if slot is None:
            entry["values"][data.values[v]] = [int(count), ord_]
        else:
            slot[0] += int(count)
            slot[1] = min(slot[1], ord_)
    numeric = kind == FIELD_NUMERIC
    for k in np.unique(key[numeric]).tolist():
        readings = data.numbers[value[numeric & (key == k)]]
        keys[data.keys[k]]["numeric"].append(readings.astype(np.float64))
    anchors = [
        (data.user_ids[u], base + (int(hi[u]) - 1) * 512 + _PROFILE_GROUP * 128)
        for u in range(len(lo))
        if hi[u] > lo[u]
    ]
    return {"keys": keys, "anchors": anchors}


def _merge_statistics(
    parts: list[dict[str, Any]],
    profiles: Mapping[str, EntityProfile],
    cutoff: datetime,
) -> FitStatistics:
    merged: dict[str, dict[str, Any]] = {}

    def entry_for(key: str) -> dict[str, Any]:
        return merged.setdefault(key, {"min_ord": None, "n": 0, "values": {}, "numeric": []})

    def add(key: str, value: str, count: int, ord_: int) -> None:
        entry = entry_for(key)
        entry["n"] += count
        if entry["min_ord"] is None or ord_ < entry["min_ord"]:
            entry["min_ord"] = ord_
        slot = entry["values"].get(value)
        if slot is None:
            entry["values"][value] = [count, ord_]
        else:
            slot[0] += count
            slot[1] = min(slot[1], ord_)

    for part in parts:
        for key, entry in part["keys"].items():
            target = entry_for(key)
            target["numeric"].extend(entry["numeric"])
            target["n"] += entry["n"]
            if target["min_ord"] is None or entry["min_ord"] < target["min_ord"]:
                target["min_ord"] = entry["min_ord"]
            for value, (count, ord_) in entry["values"].items():
                slot = target["values"].get(value)
                if slot is None:
                    target["values"][value] = [count, ord_]
                else:
                    slot[0] += count
                    slot[1] = min(slot[1], ord_)
        for user, anchor in part["anchors"]:
            profile = profiles.get(user)
            if profile is None:
                continue
            items = [(k, v) for k, v in profile.attributes.items()]
            items += [(i.key, i.key) for i in profile.lifelong if i.timestamp <= cutoff]
            if len(items) > 128:
                raise SchemaError(f"Profile {user!r} has more than 128 fit items.")
            for index, (key, value) in enumerate(items):
                add(key, value, 1, anchor + index)
    keys: dict[str, KeyFitStatistics] = {}
    for key in sorted(merged, key=lambda k: merged[k]["min_ord"]):
        entry = merged[key]
        ordered = sorted(entry["values"].items(), key=lambda kv: kv[1][1])
        numeric = np.concatenate(entry["numeric"]) if entry["numeric"] else np.zeros(0, np.float64)
        keys[key] = KeyFitStatistics(
            counts={value: count for value, (count, _) in ordered},
            n_samples=int(entry["n"]),
            numeric=numeric,
        )
    return FitStatistics(keys=keys)


def encode_block(
    root: str,
    start: int,
    stop: int,
    block: int,
    tokenizer_state: Mapping[str, Any],
    users: Collection[str] | None,
    out_dir: str,
) -> dict[str, Any]:
    """Pass 2 for one block: every event's tokens, written to `out_dir/part_{block}.npz`."""
    data = read_block(Path(root), start, stop, users=users)
    tokenizer = KeyValueTimeTokenizer.from_state(tokenizer_state)
    n_values = len(data.values)
    code = data.pair_key * n_values + data.pair_value
    uniq, inverse = np.unique(code, return_inverse=True)
    lengths = np.empty(len(uniq), np.int64)
    flat_k: list[int] = []
    flat_v: list[int] = []
    flat_p: list[int] = []
    for i, pair in enumerate(uniq.tolist()):
        k, v = divmod(pair, n_values)
        keys, values, positions = tokenizer.field_tokens(data.keys[k], data.values[v])
        lengths[i] = len(keys)
        flat_k.extend(keys)
        flat_v.extend(values)
        flat_p.extend(positions)
    uniq_start = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    key_rank = np.empty(len(data.keys), np.int64)
    key_rank[sorted(range(len(data.keys)), key=lambda i: tokenizer.field_order(data.keys[i]))] = (
        np.arange(len(data.keys))
    )

    order = np.lexsort((key_rank[data.pair_key], data.pair_event))
    pair_uniq = inverse.reshape(-1)[order]
    counts = lengths[pair_uniq]
    total = int(counts.sum())
    before = np.cumsum(counts) - counts
    gather = np.repeat(uniq_start[pair_uniq] - before, counts) + np.arange(total)
    token_event = np.repeat(data.pair_event[order], counts)
    event_counts = np.bincount(token_event, minlength=data.n_events).astype(np.int64)
    event_first = np.cumsum(event_counts) - event_counts
    rank = np.arange(total) - event_first[token_event]
    cap = tokenizer.config.max_event_tokens
    keep = np.ones(total, bool) if cap is None else rank < cap
    kept_counts = event_counts if cap is None else np.minimum(event_counts, cap)

    key_ids = np.asarray(flat_k, np.int64)[gather[keep]]
    value_ids = np.asarray(flat_v, np.int64)[gather[keep]]
    positions = np.asarray(flat_p, np.int64)[gather[keep]]
    if tokenizer.vocab_size >= 1 << 16 or (len(positions) and positions.max() >= 1 << 8):
        raise UnsupportedCombinationError(
            "The token cache stores ids as uint16 and positions as uint8; this vocabulary "
            f"({tokenizer.vocab_size}) or text length does not fit."
        )

    amount = np.zeros(data.n_events, np.float64)
    if "amount" in data.keys:
        k = data.keys.index("amount")
        is_amount = (data.pair_key == k) & (data.pair_kind == FIELD_NUMERIC)
        amount[data.pair_event[is_amount]] = data.numbers[data.pair_value[is_amount]]
    days = np.floor_divide(data.ts_us, 86_400_000_000)
    hour = np.floor_divide(data.ts_us, 3_600_000_000) % 24
    dow = (days + 3) % 7
    day = days.astype("datetime64[D]")
    dom = (day - day.astype("datetime64[M]").astype("datetime64[D]")).astype(np.int64) + 1
    by_loader = np.empty(data.n_events, np.int32)
    user_of_event = np.repeat(np.arange(len(data.user_ids)), np.diff(data.user_offsets))
    first_of_user = data.user_offsets[user_of_event]
    by_loader[first_of_user + data.loader_index] = (
        np.arange(data.n_events) - first_of_user
    ).astype(np.int32)

    path = Path(out_dir) / f"part_{block:05d}.npz"
    np.savez(
        path,
        ts_us=data.ts_us,
        id_us=data.id_us,
        hour=hour.astype(np.uint8),
        dow=dow.astype(np.uint8),
        dom=dom.astype(np.uint8),
        amount=amount,
        by_loader=by_loader,
        event_tokens=kept_counts.astype(np.int64),
        key_ids=key_ids.astype(np.uint16),
        value_ids=value_ids.astype(np.uint16),
        positions=positions.astype(np.uint8),
        user_events=np.diff(data.user_offsets).astype(np.int64),
    )
    return {
        "block": block,
        "path": str(path),
        "users": list(data.user_ids),
        "n_events": data.n_events,
        "n_tokens": int(keep.sum()),
        "n_truncated_events": int((event_counts > kept_counts).sum()),
    }


_EVENT_ARRAYS = ("ts_us", "id_us", "hour", "dow", "dom", "amount", "by_loader")
_TOKEN_ARRAYS = ("key_ids", "value_ids", "positions")


def _assemble(parts: list[dict[str, Any]], out: Path) -> dict[str, int]:
    """Concatenate part files, in block order, into one `.npy` per array."""
    loaded = [np.load(p["path"]) for p in parts]
    n_events = sum(int(p["n_events"]) for p in parts)
    n_tokens = sum(int(p["n_tokens"]) for p in parts)
    for name in _EVENT_ARRAYS + _TOKEN_ARRAYS:
        size = n_tokens if name in _TOKEN_ARRAYS else n_events
        dtype = loaded[0][name].dtype if loaded else np.int64
        target = np.lib.format.open_memmap(
            out / f"{name}.npy", mode="w+", dtype=dtype, shape=(size,)
        )
        cursor = 0
        for part in loaded:
            chunk = part[name]
            target[cursor : cursor + len(chunk)] = chunk
            cursor += len(chunk)
        target.flush()
        del target
    offsets = np.lib.format.open_memmap(
        out / "token_offsets.npy", mode="w+", dtype=np.int64, shape=(n_events + 1,)
    )
    offsets[0] = 0
    cursor, total = 1, 0
    for part in loaded:
        counts = part["event_tokens"]
        offsets[cursor : cursor + len(counts)] = total + np.cumsum(counts)
        total += int(counts.sum())
        cursor += len(counts)
    offsets.flush()
    del offsets
    user_events = np.concatenate([part["user_events"] for part in loaded]) if loaded else []
    users = [u for p in parts for u in p["users"]]
    starts = np.concatenate([[0], np.cumsum(user_events)[:-1]]).astype(np.int64)
    pq.write_table(
        pa.table(
            {
                "user_id": pa.array(users, pa.string()),
                "event_start": pa.array(starts, pa.int64()),
                "event_stop": pa.array(starts + np.asarray(user_events, np.int64), pa.int64()),
            }
        ),
        out / "users.parquet",
    )
    return {"n_events": n_events, "n_tokens": n_tokens, "n_users": len(users)}


@dataclass(frozen=True)
class CacheRequest:
    """Everything a cache is built from; its manifest records all of it."""

    root: Path
    users: tuple[str, ...] | None
    """`None` caches every user in the book."""
    fit: FitSpec
    tokenizer: KeyValueTimeConfig
    population: Mapping[str, Any]
    """How `users` was chosen, for the manifest."""

    def identity(self) -> dict[str, Any]:
        """What a run checks before reading the cache."""
        config = asdict(self.tokenizer)
        config["categorical_keys"] = list(self.tokenizer.categorical_keys)
        # Caches built before hashing existed carry neither key; keep them readable.
        if self.tokenizer.hashed_keys:
            config["hashed_keys"] = list(self.tokenizer.hashed_keys)
            config["hash_layout"] = "per_key"
        else:
            del config["hashed_keys"], config["hash_buckets"]
        if self.tokenizer.excluded_keys:
            config["excluded_keys"] = list(self.tokenizer.excluded_keys)
        else:
            del config["excluded_keys"]
        return {
            "format_version": FORMAT_VERSION,
            "book": {"root": str(self.root), **book_fingerprint(self.root)},
            "population": dict(self.population),
            "fit": self.fit.as_dict(),
            "tokenizer_config": config,
        }


def build_token_cache(
    request: CacheRequest,
    out: Path,
    *,
    workers: int = 8,
    rows_per_block: int = 1_000_000,
    progress: bool = True,
) -> dict[str, Any]:
    """Build the cache in `out` and return its manifest. `out` must not exist yet."""
    if request.tokenizer.calendar_tz != "UTC":
        raise UnsupportedCombinationError(
            "The token cache computes calendar fields in UTC; calendar_tz must be UTC."
        )
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it or choose another path.")
    staging = out.with_name(out.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "parts").mkdir(parents=True)
    started = time.perf_counter()

    def log(message: str) -> None:
        if progress:
            print(f"[token-cache {time.perf_counter() - started:7.1f}s] {message}", flush=True)

    users = None if request.users is None else frozenset(request.users)
    plan = plan_blocks(request.root, rows_per_block=rows_per_block, users=users)
    if not len(plan):
        raise SchemaError(f"{request.root} has no events to cache.")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}.")
    log(f"{plan.n_users} users, {plan.n_rows} events, {len(plan)} blocks, {workers} workers")
    blocks = list(enumerate(zip(plan.starts, plan.stops, strict=True)))
    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        if workers > 1
        else None
    )

    def run(fn: Any, jobs: list[tuple[Any, ...]]) -> list[Any]:
        if pool is None:
            return [fn(*job) for job in jobs]
        return list(pool.map(fn, *zip(*jobs, strict=True)))

    try:
        stats_parts = run(
            fit_statistics_block,
            [(str(request.root), s, e, i, request.fit, users) for i, (s, e) in blocks],
        )
        log("fit statistics collected")
        profile_path = request.root / "profiles.parquet"
        profiles: dict[str, EntityProfile] = {}
        if profile_path.exists():
            table = pq.read_table(profile_path)
            if users is not None:
                table = table.filter(
                    pc.is_in(  # type: ignore[attr-defined]
                        table.column("user_id"), value_set=pa.array(sorted(users), pa.string())
                    )
                )
            pq.write_table(table, staging / "profiles.parquet")
            profiles = {p.entity_id: p for p in load_profiles(table)}
        # A train-side sequence's cutoff is train_end; milestones dated after it are left out.
        statistics = _merge_statistics(stats_parts, profiles, ensure_utc(request.fit.train_end))
        tokenizer = KeyValueTimeTokenizer(request.tokenizer).fit_statistics(statistics)
        state = tokenizer.state_dict()
        log(f"tokenizer fit: vocabulary {tokenizer.vocab_size}, {len(tokenizer.key_vocab)} keys")
        parts = run(
            encode_block,
            [
                (str(request.root), s, e, i, state, users, str(staging / "parts"))
                for i, (s, e) in blocks
            ],
        )
    finally:
        if pool is not None:
            pool.shutdown()
    log("events encoded")
    counts = _assemble(parts, staging)
    shutil.rmtree(staging / "parts")
    (staging / "tokenizer.json").write_text(json.dumps(state), encoding="utf-8")
    manifest = {
        **request.identity(),
        **counts,
        "n_truncated_events": sum(int(p["n_truncated_events"]) for p in parts),
        "vocab_size": tokenizer.vocab_size,
        "field_kind": dict(tokenizer.field_kind),
        "build": {
            "workers": workers,
            "rows_per_block": rows_per_block,
            "blocks": len(plan),
            "seconds": round(time.perf_counter() - started, 1),
        },
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    staging.rename(out)
    log(f"done: {counts['n_tokens']} tokens in {out}")
    return manifest


__all__ = ["FORMAT_VERSION", "CacheRequest", "FitSpec", "build_token_cache"]
