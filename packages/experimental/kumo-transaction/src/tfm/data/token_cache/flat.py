# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A TabFormer train split tokenized once with a flat field tokenizer (ADR 0016).

The field tokenizer's vocabulary is fixed, so an event's tokens depend only on
its own row: the builder reads the train rows of the sampled users, maps each
row with the adapter's own `tabformer_row_event` and the tokenizer's own
`event_tokens`, and stores the result as arrays. Nothing is reimplemented, so
the cached tokens are the object path's by construction; the contract tests
check the rows and batches built from them.

A cache holds, per event and in `order_events` order within each user:
`tokens` (uint16, one column per field), `ts_us`, `amount`, `fraud`,
`fraud_available_us`, and `event_id`; `users.parquet` holds each user's event
range. Pretraining windows are cut from these arrays when a run starts and
collated as the trainer reads them (`FlatTokenCache`, `CachedFlatBatches`).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray

from tfm.contracts.batch import TransactionBatch
from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.schema import ensure_utc
from tfm.contracts.tokenizers import SequenceTokenizer
from tfm.data.adapters.paths import entity_cap, resolve_data_path, sample_seed
from tfm.data.adapters.tabformer import (
    precomputed_sample,
    precomputed_split_paths,
    precomputed_train_table,
    tabformer_label_delay,
    tabformer_row_event,
)
from tfm.data.splits.temporal import parse_duration
from tfm.data.token_cache.builders import DEFAULT_BATCH_BUILDER
from tfm.data.token_cache.formats import (
    DEFAULT_TOKEN_CACHE_FORMAT,
    PretrainRows,
    cache_path,
    register_token_cache_format,
)
from tfm.data.validation.schema import validate_event
from tfm.features.tokenizers.financial import TransactionFieldTokenizer

if TYPE_CHECKING:
    from tfm.config import RecipeConfig

FLAT_FORMAT = "flat_fields"
FLAT_FORMAT_VERSION = 1
_ROWS_FILE = "rows.arrow"


@dataclass(frozen=True)
class FlatCacheRequest:
    """Everything a flat cache is built from; its manifest records all of it."""

    root: Path
    users: tuple[object, ...]
    """The sampled users as `train.parquet` stores them (`precomputed_sample`)."""
    label_delay: timedelta
    tokenizer: TransactionFieldTokenizer
    amount_field: str
    """The numeric field stored as each event's `amount` (`data.amount_field`)."""

    def identity(self) -> dict[str, Any]:
        """What a run checks before reading the cache.

        The population is the set of users, not the cap and seed that chose it:
        event ids follow the rows of exactly these users in file order, so any
        config that samples the same users reads the same cache.
        """
        train_path = precomputed_split_paths(self.root)[0]
        names = sorted({str(user) for user in self.users})
        return {
            "format": FLAT_FORMAT,
            "format_version": FLAT_FORMAT_VERSION,
            "book": {
                "root": str(self.root),
                "train_rows": int(pq.ParquetFile(train_path).metadata.num_rows),
                "train_bytes": int(train_path.stat().st_size),
            },
            "population": {
                "adapter": "tabformer",
                "split_protocol": "precomputed",
                "n_users": len(names),
                "users_sha256": hashlib.sha256(
                    json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()
                ).hexdigest(),
                "label_delay": str(self.label_delay),
            },
            "tokenizer": dict(self.tokenizer.manifest()),
            "amount_field": self.amount_field,
        }


def _encode_block(
    rows_path: str,
    start: int,
    stop: int,
    label_delay: timedelta,
    fields: tuple[str, ...],
    amount_field: str,
) -> dict[str, np.ndarray]:
    """Rows `[start, stop)` of the staged train table, as the object path maps them."""
    tokenizer = TransactionFieldTokenizer(fields=fields)
    with pa.memory_map(rows_path) as source:
        table = pa.ipc.open_file(source).read_all().slice(start, stop - start)
        rows = table.to_pylist()
    n = len(rows)
    tokens = np.empty((n, len(fields)), np.uint16)
    ts_us = np.empty(n, np.int64)
    amount = np.empty(n, np.float64)
    has_amount = np.empty(n, bool)
    fraud = np.empty(n, np.float32)
    fraud_available_us = np.empty(n, np.int64)
    entity: list[str] = []
    event_id: list[str] = []
    for i, row in enumerate(rows):
        event = validate_event(tabformer_row_event(row, index=start + i, label_delay=label_delay))
        tokens[i] = [tokenizer.lookup(token) for token in tokenizer.event_tokens(event)]
        ts_us[i] = _us(event.event_timestamp)
        amount[i] = float(event.numeric.get(amount_field, 0.0))
        has_amount[i] = amount_field in event.numeric
        label = event.labels["fraud"]
        fraud[i] = label.value
        fraud_available_us[i] = _us(label.available_at)
        entity.append(event.entity_id)
        event_id.append(event.event_id)
    return {
        "tokens": tokens,
        "ts_us": ts_us,
        "amount": amount,
        "has_amount": has_amount,
        "fraud": fraud,
        "fraud_available_us": fraud_available_us,
        "entity": np.asarray([e.encode("ascii") for e in entity], dtype=np.bytes_),
        "event_id": np.asarray([e.encode("ascii") for e in event_id], dtype=np.bytes_),
    }


def _us(ts: datetime) -> int:
    delta = ensure_utc(ts) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def build_flat_cache(
    request: FlatCacheRequest,
    out: Path,
    *,
    workers: int = 8,
    rows_per_block: int = 500_000,
    progress: bool = True,
) -> dict[str, Any]:
    """Build the cache in `out` and return its manifest. `out` must not exist yet."""
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it or choose another path.")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}.")
    staging = out.with_name(out.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    started = time.perf_counter()

    def log(message: str) -> None:
        if progress:
            print(f"[token-cache {time.perf_counter() - started:7.1f}s] {message}", flush=True)

    users = request.users
    table = precomputed_train_table(request.root, users)
    if not table.num_rows:
        raise SchemaError(f"{request.root} has no train rows for the sampled users.")
    rows_path = staging / _ROWS_FILE
    with pa.OSFile(str(rows_path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    n_rows = table.num_rows
    del table
    bounds = [(s, min(s + rows_per_block, n_rows)) for s in range(0, n_rows, rows_per_block)]
    log(f"{len(users)} users, {n_rows} train events, {len(bounds)} blocks, {workers} workers")
    fields = tuple(request.tokenizer.fields)
    jobs = [
        (str(rows_path), s, e, request.label_delay, fields, request.amount_field) for s, e in bounds
    ]
    if workers == 1:
        parts = [_encode_block(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            parts = list(pool.map(_encode_block, *zip(*jobs, strict=True)))
    log("events encoded")
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    if not merged.pop("has_amount").any():
        raise SchemaError(
            f"No train event in {request.root} has a numeric field {request.amount_field!r}, "
            "which the raw features read as the amount; set data.amount_field to the book's "
            "money field."
        )
    # build_entity_windows groups by sorted entity id, then order_events sorts
    # each history by (timestamp, event_id).
    order = np.lexsort((merged["event_id"], merged["ts_us"], merged["entity"]))
    entity = merged.pop("entity")[order]
    for key, values in merged.items():
        np.save(staging / f"{key}.npy", values[order])
    starts = np.flatnonzero(np.r_[True, entity[1:] != entity[:-1]])
    stops = np.r_[starts[1:], len(entity)]
    pq.write_table(
        pa.table(
            {
                "user_id": pa.array(
                    [u.decode("ascii") for u in entity[starts].tolist()], pa.string()
                ),
                "event_start": pa.array(starts, pa.int64()),
                "event_stop": pa.array(stops, pa.int64()),
            }
        ),
        staging / "users.parquet",
    )
    rows_path.unlink()
    manifest = {
        **request.identity(),
        "n_users": len(starts),
        "n_events": n_rows,
        "build": {
            "workers": workers,
            "rows_per_block": rows_per_block,
            "seconds": round(time.perf_counter() - started, 1),
        },
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    staging.rename(out)
    log(f"done: {n_rows} events of {len(starts)} users in {out}")
    return manifest


@dataclass(frozen=True)
class FlatRows:
    """Sequences as `(user, [lo, hi))` event ranges, all cut at one cutoff."""

    user: NDArray[np.int64]
    lo: NDArray[np.int64]
    hi: NDArray[np.int64]
    cutoff: datetime

    def __len__(self) -> int:
        return len(self.user)

    def take(self, index: Sequence[int] | NDArray[np.int64]) -> FlatRows:
        idx = np.asarray(index, np.int64)
        return FlatRows(self.user[idx], self.lo[idx], self.hi[idx], self.cutoff)


class FlatTokenCache:
    """A built flat cache: its manifest and memory-mapped arrays."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise SchemaError(
                f"No token cache at {self.path}; build it with `tfm cache-tokens --config ...`."
            )
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("format") != FLAT_FORMAT
            or self.manifest.get("format_version") != FLAT_FORMAT_VERSION
        ):
            raise SchemaError(
                f"Token cache {self.path} is not a {FLAT_FORMAT} cache of format "
                f"{FLAT_FORMAT_VERSION}. Rebuild it with `tfm cache-tokens`."
            )
        users = pq.read_table(self.path / "users.parquet")
        self.user_ids: tuple[str, ...] = tuple(str(u) for u in users.column("user_id").to_pylist())
        self.user_start = users.column("event_start").to_numpy().astype(np.int64)
        self.user_stop = users.column("event_stop").to_numpy().astype(np.int64)

        def load(name: str) -> np.ndarray:
            return np.load(self.path / f"{name}.npy", mmap_mode="r")

        self.tokens = load("tokens")
        self.ts_us = load("ts_us")
        self.amount = load("amount")
        self.fraud = load("fraud")
        self.fraud_available_us = load("fraud_available_us")
        self.event_id = load("event_id")

    def check_identity(self, expected: Mapping[str, Any]) -> None:
        """Raise unless the cache was built from the book, sample, and tokenizer expected."""
        for name, want in expected.items():
            have = self.manifest.get(name)
            if have != want:
                raise SchemaError(
                    f"Token cache {self.path} was built with {name} {have!r}, but this run "
                    f"needs {want!r}. Rebuild it with `tfm cache-tokens`."
                )

    def train_cutoff(self) -> datetime:
        """The last cached event's time: the precomputed protocol's train cutoff."""
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=int(np.max(self.ts_us)))

    def train_windows(
        self,
        *,
        cutoff: datetime,
        lookback: timedelta | None,
        window_events: int,
        stride: int | None = None,
    ) -> FlatRows:
        """`build_entity_windows` over the cached events, as `(user, lo, hi)` ranges."""
        if window_events <= 0:
            raise ValueError(f"window_events must be positive, got {window_events}.")
        step = window_events if stride is None else stride
        if step <= 0:
            raise ValueError(f"stride must be positive, got {step}.")
        cutoff_us = _us(cutoff)
        start_us = None if lookback is None else _us(ensure_utc(cutoff) - lookback)
        user: list[int] = []
        lo: list[int] = []
        hi: list[int] = []
        for u, (first, last) in enumerate(zip(self.user_start, self.user_stop, strict=True)):
            ts = np.asarray(self.ts_us[first:last])
            keep = np.flatnonzero(
                (ts <= cutoff_us) & (True if start_us is None else ts >= start_us)
            )
            if not len(keep):
                continue
            # Kept events are contiguous: the history is in time order.
            begin, end = first + int(keep[0]), first + int(keep[-1]) + 1
            n = end - begin
            for offset in range(0, n, step):
                user.append(u)
                lo.append(begin + offset)
                hi.append(begin + min(offset + window_events, n))
                if offset + window_events >= n:
                    break
        return FlatRows(
            np.asarray(user, np.int64),
            np.asarray(lo, np.int64),
            np.asarray(hi, np.int64),
            ensure_utc(cutoff),
        )

    def collate(
        self, rows: FlatRows, tokenizer: TransactionFieldTokenizer, *, max_length: int
    ) -> TransactionBatch:
        """`collate_batch(sequences, tokenizer, max_tokens=max_length)` for these rows."""
        n_fields = len(tokenizer.fields)
        if self.tokens.shape[1] != n_fields:
            raise SchemaError(
                f"Token cache {self.path} holds {self.tokens.shape[1]} fields per event; "
                f"the tokenizer encodes {n_fields}."
            )
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        fraud: list[float] = []
        features: list[list[float]] = []
        cutoff_us = _us(rows.cutoff)
        for lo, hi in zip(rows.lo.tolist(), rows.hi.tolist(), strict=True):
            events = np.asarray(self.tokens[lo:hi], np.int64)
            body = np.full((len(events), n_fields + 1), tokenizer.sep_id, np.int64)
            body[:, :n_fields] = events
            # <bos>, each event's fields with a <sep> between events, <eos>.
            flat = [tokenizer.bos_id, *body.reshape(-1)[:-1].tolist(), tokenizer.eos_id]
            row_ids = tokenizer._truncate(flat, max_length)
            ids.append(row_ids + [tokenizer.pad_id] * (max_length - len(row_ids)))
            masks.append([1] * len(row_ids) + [0] * (max_length - len(row_ids)))
            available = np.asarray(self.fraud_available_us[lo:hi]) <= cutoff_us
            fraud.append(
                1.0 if bool((np.asarray(self.fraud[lo:hi])[available] >= 0.5).any()) else 0.0
            )
            amounts = [
                value if math.isfinite(value) else 0.0
                for value in np.asarray(self.amount[lo:hi]).tolist()
            ]
            last = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
                microseconds=int(self.ts_us[hi - 1])
            )
            features.append(
                [
                    float(sum(amounts) / len(amounts)),
                    float(max(amounts)),
                    float(len(amounts)),
                    float(last.hour),
                    float(last.weekday()),
                ]
            )
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.tensor(masks, dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        entity_ids = tuple(self.user_ids[u] for u in rows.user.tolist())
        cutoff = rows.cutoff.isoformat()
        return TransactionBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            fraud_labels=torch.tensor(fraud, dtype=torch.float32),
            numeric_features=torch.tensor(features, dtype=torch.float32).reshape(len(rows), 5),
            entity_ids=entity_ids,
            sequence_ids=tuple(f"{entity}:{cutoff}" for entity in entity_ids),
            event_ids=tuple(
                bytes(self.event_id[hi - 1]).decode("ascii") for hi in rows.hi.tolist()
            ),
        )


class CachedFlatBatches(Sequence[TransactionBatch]):
    """Minibatches of cached rows, collated when indexed.

    The same rows, in the same order and padding, as collating all of `rows`,
    `shuffled_rows(batch, shuffle_seed)`, then `minibatches(batch, batch_size)`.
    """

    def __init__(
        self,
        cache: FlatTokenCache,
        rows: FlatRows,
        tokenizer: TransactionFieldTokenizer,
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

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    @overload
    def __getitem__(self, index: int) -> TransactionBatch: ...
    @overload
    def __getitem__(self, index: slice) -> list[TransactionBatch]: ...
    def __getitem__(self, index: int | slice) -> TransactionBatch | list[TransactionBatch]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        chosen = self.order[index * self.batch_size : (index + 1) * self.batch_size]
        return self.cache.collate(
            self.rows.take(chosen), self.tokenizer, max_length=self.max_length
        )


def flat_cache_request(config: RecipeConfig, tokenizer: SequenceTokenizer) -> FlatCacheRequest:
    """The train split of the TabFormer sample `prepare_data` loads, in this tokenizer's fields."""
    if type(tokenizer) is not TransactionFieldTokenizer:
        raise UnsupportedCombinationError(
            f"A {FLAT_FORMAT} token cache holds transaction_field_tokenizer tokens; "
            f"{tokenizer.name!r} encodes differently."
        )
    root = resolve_data_path(config.data)
    users = precomputed_sample(
        root, cap=entity_cap(config.data), seed=sample_seed(config.data, config.seed)
    )
    return FlatCacheRequest(
        root=root,
        users=tuple(users),
        label_delay=tabformer_label_delay(config.data),
        tokenizer=tokenizer,
        amount_field=config.data.amount_field,
    )


class FlatFieldsCacheFormat:
    """TabFormer's precomputed train split; pretraining windows read without the book."""

    trains_without_book = True

    def config_problems(self, config: RecipeConfig) -> list[str]:
        data = config.data
        problems: list[str] = []
        if data.adapter != "tabformer":
            problems.append(
                f"a {FLAT_FORMAT} token cache reads TabFormer; data.adapter is {data.adapter!r}"
            )
        if data.split_protocol != "precomputed":
            problems.append(
                f"a {FLAT_FORMAT} token cache holds TabFormer's precomputed train split; "
                f"split_protocol is {data.split_protocol!r}"
            )
        if data.pretrain_windows is None:
            problems.append(
                f"a {FLAT_FORMAT} token cache serves pretraining windows; set data.pretrain_windows"
            )
        if config.training.packing:
            problems.append("a token cache serves padded windows; training.packing must be false")
        if data.batch_builder != DEFAULT_BATCH_BUILDER:
            problems.append(
                f"data.batch_builder reads {DEFAULT_TOKEN_CACHE_FORMAT} token caches; a "
                f"{FLAT_FORMAT} cache has its own reader, so leave it unset"
            )
        return problems

    def build(
        self,
        config: RecipeConfig,
        tokenizer: SequenceTokenizer,
        *,
        workers: int,
        rows_per_block: int,
    ) -> dict[str, Any]:
        return build_flat_cache(
            flat_cache_request(config, tokenizer),
            cache_path(config),
            workers=workers,
            rows_per_block=rows_per_block,
        )

    def profile(
        self,
        config: RecipeConfig,
        tokenizer: SequenceTokenizer,
        *,
        workers: int,
        rows_per_block: int,
        scratch: Path | None,
    ) -> dict[str, Any]:
        raise UnsupportedCombinationError(
            f"`tfm cache-tokens --profile` projects {DEFAULT_TOKEN_CACHE_FORMAT} builds; a "
            f"{FLAT_FORMAT} cache has no profile."
        )

    def open(self, config: RecipeConfig, tokenizer: SequenceTokenizer) -> FlatTokenCache:
        cache = FlatTokenCache(cache_path(config))
        cache.check_identity(flat_cache_request(config, tokenizer).identity())
        return cache

    def check_event_arrays(self, cache: FlatTokenCache, names: Sequence[str]) -> None:
        if names:
            raise SchemaError(
                f"Token cache {cache.path} is a {FLAT_FORMAT} cache, which holds no "
                f"per-event arrays; the model reads {list(names)}."
            )

    def run_tokenizer(
        self,
        cache: FlatTokenCache,
        tokenizer: SequenceTokenizer,
        config: RecipeConfig,
        *,
        users: Sequence[str],
        aliases: Mapping[str, str] | None,
    ) -> None:
        """None: the field tokenizer's vocabulary is fixed, so the run fits its own."""
        return None

    def train_cutoff(self, cache: FlatTokenCache) -> datetime:
        return cache.train_cutoff()

    def pretrain_rows(
        self, cache: FlatTokenCache, config: RecipeConfig, *, train_cutoff: datetime
    ) -> FlatRows:
        """The `data.pretrain_windows` windows of every cached user, cut at `train_cutoff`,
        which must be the cache's own."""
        windows = config.data.pretrain_windows
        if windows is None:
            raise UnsupportedCombinationError(
                f"A {FLAT_FORMAT} token cache serves pretraining windows; "
                "set data.pretrain_windows."
            )
        cached_cutoff = cache.train_cutoff()
        if cached_cutoff != train_cutoff:
            raise SchemaError(
                f"Token cache {cache.path} ends at {cached_cutoff.isoformat()}, but this run's "
                f"train cutoff is {train_cutoff.isoformat()}. Rebuild it with `tfm cache-tokens`."
            )
        return cache.train_windows(
            cutoff=train_cutoff,
            lookback=parse_duration(config.data.observation_window),
            window_events=windows.events,
        )

    def batches(
        self,
        cache: FlatTokenCache,
        rows: PretrainRows,
        tokenizer: SequenceTokenizer,
        config: RecipeConfig,
        *,
        max_length: int,
        device: torch.device,
        event_arrays: Sequence[str],
    ) -> tuple[Sequence[Any], dict[str, Any]]:
        """`CachedFlatBatches`, collated on the host; a flat cache holds no per-event
        arrays, so `event_arrays` must be empty."""
        self.check_event_arrays(cache, event_arrays)
        batches = CachedFlatBatches(
            cache,
            cast(FlatRows, rows),
            cast(TransactionFieldTokenizer, tokenizer),
            batch_size=config.training.batch_size,
            max_length=max_length,
            shuffle_seed=config.seed if config.training.shuffle else None,
        )
        return batches, {}


FLAT_FIELDS = register_token_cache_format(FLAT_FORMAT)(FlatFieldsCacheFormat())


__all__ = [
    "FLAT_FIELDS",
    "FLAT_FORMAT",
    "FLAT_FORMAT_VERSION",
    "CachedFlatBatches",
    "FlatCacheRequest",
    "FlatFieldsCacheFormat",
    "FlatRows",
    "FlatTokenCache",
    "build_flat_cache",
    "flat_cache_request",
]
