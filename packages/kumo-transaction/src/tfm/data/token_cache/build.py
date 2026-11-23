# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a token cache from a `ColumnarSource` through a `ColumnarTokenizer`.

Two passes over blocks of whole users, each block in its own process:

1. **Fit statistics.** Each block is cut to the train-side sequences `prepare_data`
   would fit on (events in the observation window up to the train cutoff, the most
   recent `max_events`), and `fit_block` collects its statistics. `merge` combines
   them in the order blocks finish; the tokenizer is then fit from the result alone.
2. **Encode.** `encode_block` gives every event of every user, at every time, its
   tokens. An event's tokens depend only on its own fields, so any later sequence is
   a slice of the cache. The build adds the per-event arrays every cache holds
   (`ENGINE_ARRAYS`) to the tokenizer's own.

No label is read. Events after the train cutoff are encoded but never counted.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import itertools
import json
import pickle
import shutil
import struct
import sys
import time
import warnings
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from tfm.contracts.errors import SchemaError, TfmError, UnsupportedCombinationError
from tfm.contracts.schema import EntityProfile, ensure_utc
from tfm.contracts.sources import ColumnarSource, PlannedBlock
from tfm.contracts.tokenizers import (
    FIELD_NUMERIC,
    ColumnarTokenizer,
    EventBlock,
    event_array_problem,
)
from tfm.data.adapters.pragmatiq import load_profiles
from tfm.data.validation.schema import validate_event_block
from tfm.features.tokenizers.registry import tokenizer_class

FORMAT_VERSION = 4
MAX_PARTS = 256
"""A cache's blocks are written as at most this many runs of consecutive blocks, one file each."""
_ALIGN = 64
PART_MAGIC = b"TFMPART\0"


def part_header(build_id: str, part: int) -> bytes:
    """The 32 bytes that open part file `part`: magic, format version, part, build id."""
    return PART_MAGIC + struct.pack("<II", FORMAT_VERSION, part) + bytes.fromhex(build_id)


ENGINE_ARRAYS = ("ts_us", "id_us", "amount", "by_loader")
"""Per-event arrays every cache holds, beside the tokenizer's own."""


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
    a = user_offsets[:-1].astype(np.int64)

    def before(t: int) -> NDArray[np.int64]:
        """Per user, the first event at or after `t`; each user's events are time-sorted."""
        below = np.concatenate([[0], np.cumsum(ts_us < t)])
        return a + below[user_offsets[1:]] - below[user_offsets[:-1]]

    hi = before(end)
    lo = (
        a
        if lookback is None
        else np.minimum(before(end - lookback // timedelta(microseconds=1)), hi)
    )
    if max_events > 0:
        lo = np.maximum(lo, hi - max_events)
    return lo.astype(np.int64), hi.astype(np.int64)


@dataclass(frozen=True)
class BlockStatistics[StatsT]:
    """One block's fit statistics, and whether a train-side event carries the amount."""

    statistics: StatsT
    has_amount: bool


def _train_side(block: EventBlock, fit: FitSpec) -> EventBlock:
    """`block` cut to the events each user's train-side sequence keeps, in order."""
    lo, hi = train_ranges(
        block.ts_us,
        block.user_offsets,
        train_end=fit.train_end,
        lookback=fit.observation_window,
        max_events=fit.max_events,
    )
    counts = hi - lo
    starts = np.cumsum(counts) - counts
    picked = np.repeat(lo - starts, counts) + np.arange(int(counts.sum()))
    in_fit = np.zeros(block.n_events, bool)
    in_fit[picked] = True
    renumber = np.cumsum(in_fit) - 1
    mask = in_fit[block.pair_event]
    # Only the keys and values the kept pairs use, so no later field or value reaches the fit.
    keys, pair_key = np.unique(block.pair_key[mask], return_inverse=True)
    values, pair_value = np.unique(block.pair_value[mask], return_inverse=True)
    return dataclasses.replace(
        block,
        user_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        ts_us=block.ts_us[picked],
        id_us=block.id_us[picked],
        loader_index=block.loader_index[picked],
        pair_event=renumber[block.pair_event[mask]],
        pair_key=pair_key.astype(np.int64),
        pair_value=pair_value.astype(np.int64),
        pair_kind=block.pair_kind[mask],
        pair_map_position=block.pair_map_position[mask],
        keys=tuple(block.keys[k] for k in keys.tolist()),
        values=tuple(block.values[v] for v in values.tolist()),
        numbers=block.numbers[values],
    )


def fit_statistics_block[StatsT](
    block: EventBlock,
    tokenizer: ColumnarTokenizer[StatsT],
    fit: FitSpec,
    profiles: Path | None,
    amount_field: str,
) -> BlockStatistics[StatsT]:
    """Pass 1 for one block: `tokenizer.fit_block` of its train-side events.

    The block's users' profiles, read from `profiles` (the cache's `profiles.parquet`,
    as `ColumnarSource.write_profiles` writes it), are passed as of `fit.train_end` when
    the tokenizer uses profiles. A block with a label-like input field is refused.
    """
    validate_event_block(block)
    train = _train_side(block, fit)
    owned: dict[str, EntityProfile] = {}
    if tokenizer.uses_profiles:
        fitted = np.flatnonzero(np.diff(train.user_offsets)).tolist()
        cutoff = ensure_utc(fit.train_end)
        # A train-side sequence's cutoff is train_end; milestones dated after it are left out.
        owned = {
            user: dataclasses.replace(
                p, lifelong=tuple(i for i in p.lifelong if i.timestamp <= cutoff)
            )
            for user, p in _block_profiles(profiles, [train.user_ids[u] for u in fitted]).items()
        }
    has_amount = amount_field in train.keys and bool(
        (
            (train.pair_key == train.keys.index(amount_field)) & (train.pair_kind == FIELD_NUMERIC)
        ).any()
    )
    return BlockStatistics(tokenizer.fit_block(train, owned), has_amount)


def _fit_job[StatsT](
    source: ColumnarSource[Any],
    planned: PlannedBlock,
    users: Collection[str] | None,
    tokenizer: ColumnarTokenizer[StatsT],
    fit: FitSpec,
    profiles: Path | None,
    amount_field: str,
) -> BlockStatistics[StatsT]:
    return fit_statistics_block(
        source.read_block(planned, users=users), tokenizer, fit, profiles, amount_field
    )


def _block_profiles(path: Path | None, user_ids: list[str]) -> dict[str, EntityProfile]:
    """The profiles of `user_ids` (ascending), read from the row groups whose `user_id`
    range can hold them; a later row wins over an earlier one for the same user."""
    if path is None or not user_ids:
        return {}
    file = pq.ParquetFile(path)
    column = file.schema_arrow.get_field_index("user_id")
    first, last = user_ids[0], user_ids[-1]
    groups = []
    for g in range(file.num_row_groups):
        stats = file.metadata.row_group(g).column(column).statistics
        if stats is None or not stats.has_min_max or (stats.max >= first and stats.min <= last):
            groups.append(g)
    if not groups:
        return {}
    table = file.read_row_groups(groups)
    wanted = pa.array(user_ids, pa.string())
    table = table.filter(pc.is_in(table.column("user_id"), value_set=wanted))  # type: ignore[attr-defined]
    return {p.entity_id: p for p in load_profiles(table)}


def _completed(
    pool: ProcessPoolExecutor | None,
    fn: Callable[..., Any],
    jobs: Iterable[tuple[Any, ...]],
    limit: int,
) -> Iterator[Any]:
    """`fn(*job)` for every job, yielded as each finishes, at most `limit` at a time."""
    if pool is None:
        yield from (fn(*job) for job in jobs)
        return
    todo = iter(jobs)
    running: set[Future[Any]] = {pool.submit(fn, *job) for job in itertools.islice(todo, limit)}
    while running:
        done, running = wait(running, return_when=FIRST_COMPLETED)
        running |= {pool.submit(fn, *job) for job in itertools.islice(todo, len(done))}
        yield from (future.result() for future in done)


def _same_arrays(want: Mapping[str, str], got: Mapping[str, str]) -> None:
    if list(got.items()) != list(want.items()):
        raise SchemaError(
            f"A block's arrays {dict(got)} differ from the first block's {dict(want)}; "
            "every block must return the same per-event array names and dtypes."
        )


def _block_arrays(
    block: EventBlock, tokenizer: ColumnarTokenizer[Any], amount_field: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """One block's arrays, in the order a part stores them, and its totals."""
    encoded = tokenizer.encode_block(block)
    reserved = {*ENGINE_ARRAYS, "token_offsets", "key_ids", "value_ids", "positions"}
    if taken := sorted(reserved & set(encoded.events)):
        raise SchemaError(
            f"The tokenizer's per-event arrays {taken} use names the cache keeps for its own "
            "arrays; rename them."
        )
    for name, a in encoded.events.items():
        if a.shape != (block.n_events,):
            raise SchemaError(
                f"Per-event array {name!r} has shape {a.shape}; it needs one value per event, "
                f"({block.n_events},)."
            )
        if problem := event_array_problem(a.dtype):
            raise SchemaError(f"Per-event array {name!r} {problem}.")
    amount = np.zeros(block.n_events, np.float64)
    if amount_field in block.keys:
        k = block.keys.index(amount_field)
        is_amount = (block.pair_key == k) & (block.pair_kind == FIELD_NUMERIC)
        amount[block.pair_event[is_amount]] = block.numbers[block.pair_value[is_amount]]
    by_loader = np.empty(block.n_events, np.int32)
    user_of_event = np.repeat(np.arange(len(block.user_ids)), np.diff(block.user_offsets))
    first_of_user = block.user_offsets[user_of_event]
    by_loader[first_of_user + block.loader_index] = (
        np.arange(block.n_events) - first_of_user
    ).astype(np.int32)
    arrays = {
        "ts_us": block.ts_us,
        "id_us": block.id_us,
        **encoded.events,
        "amount": amount,
        "by_loader": by_loader,
        "token_offsets": encoded.token_offsets,
        "key_ids": encoded.key_ids,
        "value_ids": encoded.value_ids,
        "positions": encoded.positions,
    }
    totals = {
        "events": block.n_events,
        "tokens": len(encoded.key_ids),
        "users": len(block.user_ids),
        "truncated_events": encoded.truncated_events,
    }
    return arrays, totals


def encode_part(
    source: ColumnarSource[Any],
    blocks: Sequence[PlannedBlock],
    tokenizer_type: type[ColumnarTokenizer[Any]],
    tokenizer_state: Mapping[str, Any],
    users: Collection[str] | None,
    out_dir: str,
    part: int,
    build_id: str,
    amount_field: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Pass 2 for consecutive blocks, written after `part_header` to `out_dir/parts/{part}.bin`.

    Returns each block's totals, part and byte offset of each array, and each array's
    dtype; each block's users go to `out_dir/users/{index}.parquet`.
    """
    tokenizer = tokenizer_type.from_state(tokenizer_state)
    out = Path(out_dir)
    stats = []
    dtypes: dict[str, str] | None = None
    with open(out / "parts" / f"{part:05d}.bin", "wb") as file:
        file.write(part_header(build_id, part))
        for planned in blocks:
            block = source.read_block(planned, users=users)
            arrays, totals = _block_arrays(block, tokenizer, amount_field)
            got = {name: np.asarray(a).dtype.str for name, a in arrays.items()}
            if dtypes is None:
                dtypes = got
            _same_arrays(dtypes, got)
            offsets = {}
            for name, array in arrays.items():
                file.write(bytes(-file.tell() % _ALIGN))
                offsets[name] = file.tell()
                np.asarray(array).tofile(file)
            table = pa.table(
                {
                    "user_id": pa.array(block.user_ids, pa.string()),
                    "n_events": pa.array(np.diff(block.user_offsets), pa.int64()),
                }
            )
            pq.write_table(table, out / "users" / f"{planned.index:05d}.parquet")
            stats.append({**totals, "part": part, "offsets": offsets})
    return stats, dtypes or {}


def _write_users(folder: Path, n_blocks: int, out: Path) -> int:
    """Every block's users, in block order, with their event ranges in the whole cache.

    Returns the longest user id's length in bytes, and warns when it is over four times
    the mean: a reader stores every id at that length.
    """
    writer: pq.ParquetWriter | None = None
    first = n_ids = total_bytes = 0
    longest: tuple[int, str] = (0, "")
    for i in range(n_blocks):
        path = folder / f"{i:05d}.parquet"
        users = pq.read_table(path)
        stop = first + np.cumsum(users.column("n_events").to_numpy().astype(np.int64))
        table = pa.table(
            {
                "user_id": users.column("user_id"),
                "event_start": pa.array(stop - users.column("n_events").to_numpy(), pa.int64()),
                "event_stop": pa.array(stop, pa.int64()),
            }
        )
        if writer is None:
            writer = pq.ParquetWriter(out, table.schema)
        writer.write_table(table)
        lengths = pc.binary_length(users.column("user_id")).to_numpy()  # type: ignore[attr-defined]
        if len(lengths):
            n_ids, total_bytes = n_ids + len(lengths), total_bytes + int(lengths.sum())
            j = int(lengths.argmax())
            if lengths[j] > longest[0]:
                longest = (int(lengths[j]), users.column("user_id")[j].as_py())
        first = int(stop[-1]) if len(stop) else first
        path.unlink()
    if writer is not None:
        writer.close()
    if n_ids and longest[0] > 4 * total_bytes / n_ids:
        warnings.warn(
            f"User id {longest[1][:40]!r}... is {longest[0]} bytes, over four times the mean "
            f"of {total_bytes / n_ids:.1f}. A token cache stores every user id at the longest "
            "one's length, so this id sets the memory of the whole user table; check the "
            "book's user_id column for a malformed row.",
            stacklevel=3,
        )
    return longest[0]


@dataclass(frozen=True)
class CacheRequest:
    """Everything a cache is built from; its manifest records all of it."""

    source: ColumnarSource[Any]
    users: tuple[str, ...] | None
    """`None` caches every user in the book."""
    fit: FitSpec
    tokenizer: ColumnarTokenizer[Any]
    """Unfitted; the build fits a copy."""
    population: Mapping[str, Any]
    """How `users` was chosen, for the manifest."""
    amount_field: str
    """The numeric field stored as each event's `amount` (`data.amount_field`)."""

    def identity(self) -> dict[str, Any]:
        """What a run checks before reading the cache."""
        cls = type(self.tokenizer)
        population = dict(self.population)
        if self.users is not None:
            names = json.dumps(sorted(set(self.users)), ensure_ascii=False, separators=(",", ":"))
            population["users_sha256"] = hashlib.sha256(names.encode()).hexdigest()
        return {
            "format_version": FORMAT_VERSION,
            "book": self.source.fingerprint(),
            "population": population,
            "fit": self.fit.as_dict(),
            "tokenizer": f"{cls.__module__}:{cls.__qualname__}",
            "tokenizer_settings": self.tokenizer.settings(),
            "amount_field": self.amount_field,
        }


def _preflight(
    request: CacheRequest, planned: PlannedBlock, profiles: Path | None, *, workers: int
) -> None:
    """Run one block through fit_block, merge, fit_stats, a JSON state round trip and
    encode, and pickle what workers receive, so a source or tokenizer that cannot build
    fails before the build starts. The block's two halves of users, merged in either
    order, must fit the state the whole block fits."""
    users = None if request.users is None else frozenset(request.users)
    parts: list[object] = [request.source, request.tokenizer]
    for part in parts:
        if type(part).__module__ == "__main__" and not hasattr(sys.modules["__main__"], "__file__"):
            raise UnsupportedCombinationError(
                f"{type(part).__qualname__} is defined in an interactive session, so worker "
                "processes cannot load it; define it in an importable module."
            )
    cls = type(request.tokenizer)
    if tokenizer_class(f"{cls.__module__}:{cls.__qualname__}") is not cls:
        raise UnsupportedCombinationError(
            f"{cls.__module__}:{cls.__qualname__} imports another class than the tokenizer; "
            "define it once, at the top level of an importable module."
        )
    try:
        if workers > 1:
            for part in parts:
                pickle.loads(pickle.dumps(part))
        block = request.source.read_block(planned, users=users)
        tokenizer = copy.deepcopy(request.tokenizer)
        json.dumps(tokenizer.settings())
        stats = fit_statistics_block(
            block, tokenizer, request.fit, profiles, request.amount_field
        ).statistics
        if len(block.user_ids) > 1:
            _check_split(request, planned, block, stats, profiles)
        fitted = tokenizer.fit_stats(tokenizer.merge(stats, copy.deepcopy(stats)))
        state = json.loads(json.dumps(fitted.state_dict()))
        restored = type(fitted).from_state(state)
        if restored.state_dict() != state:
            raise SchemaError("from_state(state).state_dict() differs from state.")
        _block_arrays(block, restored, request.amount_field)
    except TfmError:
        raise
    except Exception as error:
        raise UnsupportedCombinationError(
            f"{type(request.tokenizer).__qualname__} on {request.source} failed the pre-build "
            f"check on block {planned.index}: {error!r}. Check it with "
            "`tfm.testing.conformance.check_columnar_tokenizer`."
        ) from error


def _check_split(
    request: CacheRequest,
    planned: PlannedBlock,
    block: EventBlock,
    stats: object,
    profiles: Path | None,
) -> None:
    half = len(block.user_ids) // 2
    first = request.source.read_block(planned, users=frozenset(block.user_ids[:half]))
    second = request.source.read_block(planned, users=frozenset(block.user_ids[half:]))
    # Numbered as a plan with half the rows per block would number them.
    second = dataclasses.replace(second, index=planned.index + 1)
    a, b = (
        fit_statistics_block(
            part, request.tokenizer, request.fit, profiles, request.amount_field
        ).statistics
        for part in (first, second)
    )

    def state(parts: list[object]) -> object:
        tokenizer = copy.deepcopy(request.tokenizer)
        merged = functools.reduce(tokenizer.merge, copy.deepcopy(parts))
        return tokenizer.fit_stats(merged).state_dict()

    whole, halves = state([stats]), state([a, b])
    if halves != state([b, a]):
        raise UnsupportedCombinationError(
            f"{type(request.tokenizer).__qualname__}: merge order changes the fitted state "
            f"on block {planned.index}, so the cache would depend on the worker count. "
            "Check it with `tfm.testing.conformance.check_columnar_tokenizer`."
        )
    if halves != whole:
        raise UnsupportedCombinationError(
            f"{type(request.tokenizer).__qualname__}: fitting block {planned.index} in two "
            "halves gives another state than fitting it whole, so the cache would depend on "
            "rows_per_block. Check it with `tfm.testing.conformance.check_columnar_tokenizer`."
        )


def build_token_cache(
    request: CacheRequest,
    out: Path,
    *,
    workers: int = 8,
    rows_per_block: int = 1_000_000,
    progress: bool = True,
) -> dict[str, Any]:
    """Build the cache in `out` and return its manifest. `out` must not exist yet.

    The first block runs through every step once, in this process, before the fit pass:
    a source or tokenizer that cannot build fails there."""
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it or choose another path.")
    staging = out.with_name(out.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "parts").mkdir(parents=True)
    (staging / "users").mkdir()
    started = time.perf_counter()

    def log(message: str) -> None:
        if progress:
            print(f"[token-cache {time.perf_counter() - started:7.1f}s] {message}", flush=True)

    users = None if request.users is None else frozenset(request.users)
    plan = request.source.plan(rows_per_block=rows_per_block, users=users)
    if not plan.blocks:
        raise SchemaError(f"{request.source} has no events to cache.")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}.")
    log(f"{plan.n_users} users, {plan.n_rows} events, {len(plan.blocks)} blocks, {workers} workers")
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
        profiles: Path | None = staging / "profiles.parquet"
        if not request.source.write_profiles(users, staging / "profiles.parquet"):
            profiles = None
        _preflight(request, plan.blocks[0], profiles, workers=workers)
        log("preflight passed")
        tokenizer = copy.deepcopy(request.tokenizer)
        fit_jobs = (
            (request.source, b, users, tokenizer, request.fit, profiles, request.amount_field)
            for b in plan.blocks
        )
        merged: Any = None
        has_amount = False
        for part in _completed(pool, _fit_job, fit_jobs, 2 * workers):
            merged = part.statistics if merged is None else tokenizer.merge(merged, part.statistics)
            has_amount |= part.has_amount
        log("fit statistics collected")
        if not has_amount:
            raise SchemaError(
                f"No train-side event in {request.source} has a numeric field "
                f"{request.amount_field!r}, which the raw features read as the amount; set "
                "data.amount_field to the book's money field."
            )
        fitted = tokenizer.fit_stats(merged)
        state = fitted.state_dict()
        log(f"tokenizer fit: vocabulary {fitted.vocab_size}")
        # Same request and blocks, same id: identical builds stay byte-identical.
        build_id = hashlib.sha256(
            json.dumps([request.identity(), rows_per_block], sort_keys=True).encode()
        ).hexdigest()[:32]
        runs = np.array_split(np.arange(len(plan.blocks)), min(len(plan.blocks), MAX_PARTS))
        encoded = run(
            encode_part,
            [
                (
                    request.source,
                    [plan.blocks[i] for i in r.tolist()],
                    type(fitted),
                    state,
                    users,
                    str(staging),
                    p,
                    build_id,
                    request.amount_field,
                )
                for p, r in enumerate(runs)
            ],
        )
        arrays = encoded[0][1]
        for _, dtypes in encoded:
            _same_arrays(arrays, dtypes)
        blocks = [s for part, _ in encoded for s in part]
    finally:
        if pool is not None:
            pool.shutdown()
    log("events encoded")
    longest_id = _write_users(staging / "users", len(blocks), staging / "users.parquet")
    (staging / "users").rmdir()
    (staging / "tokenizer.json").write_text(json.dumps(state), encoding="utf-8")
    manifest = {
        **request.identity(),
        "n_events": sum(s["events"] for s in blocks),
        "n_tokens": sum(s["tokens"] for s in blocks),
        "n_users": sum(s["users"] for s in blocks),
        "user_id_bytes": longest_id,
        "n_truncated_events": sum(s.pop("truncated_events") for s in blocks),
        "arrays": arrays,
        "parts": len(runs),
        "build_id": build_id,
        "blocks": blocks,
        "vocab_size": fitted.vocab_size,
        "build": {
            "workers": workers,
            "rows_per_block": rows_per_block,
            "blocks": len(plan.blocks),
            "seconds": round(time.perf_counter() - started, 1),
        },
    }
    if not manifest["n_tokens"]:
        shutil.rmtree(staging)
        raise SchemaError(
            f"{type(request.tokenizer).__qualname__} encoded no token for any event; a cache "
            "needs at least one."
        )
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    staging.rename(out)
    log(f"done: {manifest['n_tokens']} tokens in {out}")
    return manifest


__all__ = [
    "FORMAT_VERSION",
    "CacheRequest",
    "FitSpec",
    "build_token_cache",
    "BlockStatistics",
    "fit_statistics_block",
    "part_header",
]
