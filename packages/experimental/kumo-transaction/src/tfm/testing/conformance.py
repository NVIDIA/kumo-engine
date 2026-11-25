# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks a user component must pass to take a slot's accelerated path."""

from __future__ import annotations

import copy
import dataclasses
import functools
import gc
import json
import pickle
import tempfile
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from tfm.contracts.errors import TfmError
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.schema import TransactionSequence
from tfm.contracts.sources import ColumnarSource
from tfm.contracts.tokenizers import (
    ColumnarTokenizer,
    EncodedBlock,
    SequenceTokenizer,
    event_array_problem,
)
from tfm.data.token_cache import (
    CachedBatches,
    CacheRequest,
    DeviceBatches,
    FitSpec,
    TokenCache,
    build_token_cache,
    fit_statistics_block,
)

EVENT_FIELDS = (
    "key_ids",
    "value_ids",
    "positions",
    "token_mask",
    "event_mask",
    "time_log",
    "elapsed_log",
    "events",
    "entity_ids",
    "sequence_ids",
)
"""The batch fields a user's events decide; profiles, labels and raw features are the
run's."""


class ConformanceError(TfmError):
    """A component breaks its slot's contract."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ConformanceError(message)


def check_columnar_tokenizer(
    tokenizer: ColumnarTokenizer[Any],
    source: ColumnarSource[Any],
    *,
    fit: FitSpec,
    rows_per_block: int,
    amount_field: str = "amount",
    sequences: Sequence[TransactionSequence] | None = None,
    max_length: int = 64,
    batch_size: int = 4,
) -> None:
    """Raise `ConformanceError` unless unfitted `tokenizer`, on `source`, builds a cache
    that does not depend on block size or merge order and reads back in fixed shapes:

    - the fitted state and every event's tokens equal between blocks of
      `rows_per_block` rows (pick one that gives several) and one block;
    - each per-event array holds one number per event, in a dtype torch copies;
    - the settings and state survive JSON and pickle; `from_state` restores the state;
    - the cache's device batches (CUDA when available) all have one shape, but for the
      last batch's row count, and equal its host batches;
    - with `sequences`, the train-side sequences `fit` describes, a tokenizer that also
      implements `SequenceTokenizer` collates them to the cache's `EVENT_FIELDS`.
    """
    _require(isinstance(tokenizer, ColumnarTokenizer), "does not implement ColumnarTokenizer")
    settings = tokenizer.settings()
    _require(json.loads(json.dumps(settings)) == settings, "settings() is not JSON")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        profiles: Path | None = Path(tmp) / "profiles.parquet"
        if not source.write_profiles(None, Path(tmp) / "profiles.parquet"):
            profiles = None
        split = source.plan(rows_per_block=rows_per_block).blocks
        whole = source.plan(rows_per_block=2**62).blocks
        _require(len(split) > 1, f"rows_per_block {rows_per_block} gives one block; lower it")

        def stats(blocks: Any) -> list[Any]:
            return [
                fit_statistics_block(
                    source.read_block(b), tokenizer, fit, profiles, amount_field
                ).statistics
                for b in blocks
            ]

        def fitted(parts: list[Any]) -> ColumnarTokenizer[Any]:
            merged = functools.reduce(tokenizer.merge, copy.deepcopy(parts))
            return copy.deepcopy(tokenizer).fit_stats(merged)

        parts = stats(split)
        state = fitted(parts).state_dict()
        _require(fitted(parts[::-1]).state_dict() == state, "merge order changes the fitted state")
        _require(fitted(stats(whole)).state_dict() == state, "block size changes the fitted state")
        _require(json.loads(json.dumps(state)) == state, "state_dict() is not JSON")
        restored = type(tokenizer).from_state(state)
        _require(restored.state_dict() == state, "from_state(state).state_dict() != state")
        restored = pickle.loads(pickle.dumps(restored))
        encoded = [restored.encode_block(source.read_block(b)) for b in split]
        for e in encoded:
            _check_event_arrays(e)
        a = _events(encoded)
        b = _events([restored.encode_block(source.read_block(b)) for b in whole])
        _require(a.keys() == b.keys(), f"per-event arrays differ: {sorted(a)} vs {sorted(b)}")
        for name in a:
            _require(a[name].dtype == b[name].dtype, f"{name} dtype depends on block size")
            _require(a[name].tobytes() == b[name].tobytes(), f"{name} depends on block size")

        request = CacheRequest(
            source=source,
            users=None,
            fit=fit,
            tokenizer=tokenizer,
            population={"users": "all"},
            amount_field=amount_field,
        )
        out = Path(tmp) / "cache"
        build_token_cache(request, out, workers=1, rows_per_block=rows_per_block, progress=False)
        failure = None
        try:
            _check_cache(
                out,
                fit=fit,
                restored=restored,
                sequences=sequences,
                max_length=max_length,
                batch_size=batch_size,
            )
        except ConformanceError as error:
            failure = str(error)
        # The checks' memory maps go with their frame, which a raised error's traceback
        # keeps alive; collect them before the directory is removed, since NFS cannot
        # remove a directory whose files are still mapped.
        gc.collect()
        if failure is not None:
            raise ConformanceError(failure)


def _check_cache(
    out: Path,
    *,
    fit: FitSpec,
    restored: ColumnarTokenizer[Any],
    sequences: Sequence[TransactionSequence] | None,
    max_length: int,
    batch_size: int,
) -> None:
    """The fixed-shape, device == host and (with `sequences`) parity checks on the cache
    built in `out`."""
    cache = TokenCache(out)
    reader = cache.tokenizer(max_events=fit.max_events)
    rows = cache.train_rows(
        train_end=fit.train_end, lookback=fit.observation_window, max_events=fit.max_events
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "max_length": max_length,
        "shuffle_seed": 0,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    host = CachedBatches(cache, rows, reader, **kwargs)
    on_device = DeviceBatches(cache, rows, reader, device=device, **kwargs)
    _require(len(host) > 1, f"batch_size {batch_size} gives one batch; lower it")
    _check_fixed_shapes(on_device, host)
    if sequences is not None:
        got = cache.collate(rows, reader, max_length=max_length)
        by_user = {s.entity_id: s for s in sequences}
        _require(
            sorted(by_user) == sorted(got.entity_ids),
            "sequences are not the cache's train-side users",
        )
        want = cast(SequenceTokenizer, restored).collate(
            [by_user[u] for u in got.entity_ids], max_length=max_length
        )
        if not isinstance(want, KeyValueTimeBatch):
            raise ConformanceError(f"collate gives {type(want).__name__}, not a KeyValueTimeBatch")
        _check_equal(got, want, "encode_block and encode", EVENT_FIELDS)


def _check_event_arrays(encoded: EncodedBlock) -> None:
    """Each per-event array holds one number per event, in a dtype torch copies as is."""
    n_events = len(encoded.token_offsets) - 1
    for name, array in encoded.events.items():
        _require(
            array.shape == (n_events,),
            f"{name} has shape {array.shape}; a per-event array holds one value per event "
            f"({n_events},)",
        )
        _require(array.dtype.kind in "biuf", f"{name} has dtype {array.dtype}, not a number")
        problem = event_array_problem(array.dtype)
        _require(problem is None, f"{name} {problem}")


def _check_fixed_shapes(
    on_device: Sequence[KeyValueTimeBatch], host: Sequence[KeyValueTimeBatch]
) -> None:
    """Every device batch has batch 0's shapes, but for the last batch's row count, and
    equals its host batch."""
    _require(len(on_device) == len(host), "device and host give different batch counts")
    first = _shapes(on_device[0])
    last = len(on_device) - 1
    for i, (d, h) in enumerate(zip(on_device, host, strict=True)):
        shapes = _shapes(d)
        if i == last:
            shapes = {
                k: (first[k][0], *s[1:]) if k in first and s else s for k, s in shapes.items()
            }
        _require(shapes == first, f"device batch {i} has shapes {shapes}, batch 0 {first}")
        _check_equal(d.to(torch.device("cpu")), h, f"device batch {i} and host batch")


def _shapes(batch: KeyValueTimeBatch) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for f in dataclasses.fields(batch):
        value = getattr(batch, f.name)
        if isinstance(value, torch.Tensor):
            out[f.name] = tuple(value.shape)
        elif isinstance(value, dict):
            out.update({f"{f.name}.{k}": tuple(v.shape) for k, v in value.items()})
    return out


def _check_equal(
    got: KeyValueTimeBatch,
    want: KeyValueTimeBatch,
    what: str,
    fields: Collection[str] | None = None,
) -> None:
    for name in (f.name for f in dataclasses.fields(want)):
        if fields is not None and name not in fields:
            continue
        a, b = getattr(got, name), getattr(want, name)
        if isinstance(b, dict):
            _require(a.keys() == b.keys(), f"{what}: {name} differ: {sorted(a)} vs {sorted(b)}")
            pairs = [(f"{name}.{k}", a[k], b[k]) for k in b]
        else:
            pairs = [(name, a, b)]
        for label, x, y in pairs:
            if isinstance(y, torch.Tensor):
                same = (
                    isinstance(x, torch.Tensor)
                    and x.shape == y.shape
                    and x.dtype == y.dtype
                    and torch.equal(x, y)
                )
            else:
                same = x == y
            _require(same, f"{what}: {label} differ")


def _events(encoded: list[EncodedBlock]) -> dict[str, np.ndarray]:
    """Every event's tokens and per-event arrays, as if one block."""
    return {
        "tokens_per_event": np.concatenate([np.diff(e.token_offsets) for e in encoded]),
        "key_ids": np.concatenate([e.key_ids for e in encoded]),
        "value_ids": np.concatenate([e.value_ids for e in encoded]),
        "positions": np.concatenate([e.positions for e in encoded]),
        **{name: np.concatenate([e.events[name] for e in encoded]) for name in encoded[0].events},
    }


__all__ = ["EVENT_FIELDS", "ConformanceError", "check_columnar_tokenizer"]
