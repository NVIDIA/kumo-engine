# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collate cached rows on the device that trains on them."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import torch

from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.schema import ensure_utc
from tfm.data.token_cache.store import (
    CachedBatches,
    CachedTokenizer,
    CacheRows,
    TokenCache,
    profile_tensors,
)
from tfm.features.tokenizers.key_value_time import PAD

_INDEXABLE = {np.dtype(np.uint16): np.int32, np.dtype(np.uint32): np.int64}
"""A per-event dtype torch cannot index on a device, and the dtype it widens to."""


def _time_encode(delta_us: torch.Tensor) -> torch.Tensor:
    """`time_encode` of microsecond gaps, in float64 as the tokenizer computes it."""
    return (8.0 * torch.log1p((delta_us.double() / 1e6).clamp_min(0.0) / 8.0)).float()


class DeviceBatches(CachedBatches):
    """`CachedBatches` collated on `device` from arrays held there; only the per-event
    arrays `event_arrays` (default: all) are copied there and gathered."""

    def __init__(
        self,
        cache: TokenCache,
        rows: CacheRows,
        tokenizer: CachedTokenizer,
        *,
        batch_size: int,
        max_length: int,
        shuffle_seed: int | None,
        device: torch.device,
        event_arrays: Collection[str] | None = None,
    ) -> None:
        if (rows.hi <= rows.lo).any():
            raise UnsupportedCombinationError(
                "DeviceBatches needs every row to hold at least one event. Drop the empty "
                "rows with `rows.take(np.flatnonzero(rows.hi > rows.lo))`, or set "
                "`data.token_cache_on_device: false`."
            )
        super().__init__(
            cache,
            rows,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle_seed=shuffle_seed,
            event_arrays=event_arrays,
        )
        read = cache.events if event_arrays is None else event_arrays
        self.read_events = tuple(name for name in cache.events if name in read)
        # torch indexes no uint16, so the ids widen to int32 and per-event arrays by `_INDEXABLE`.
        arrays = {
            "token_offsets": (cache.token_offsets, None),
            "key_ids": (cache.key_ids, np.int32),
            "value_ids": (cache.value_ids, np.int32),
            "positions": (cache.positions, None),
            "id_us": (cache.id_us, None),
            **{
                name: (cache.events[name], _INDEXABLE.get(cache.events[name].dtype))
                for name in self.read_events
            },
        }
        users, row_user = np.unique(rows.user, return_inverse=True)
        profiles = profile_tensors(tokenizer, cache.user_ids(users), rows.cutoff, self.pad_to[2])
        raw_features = cache.raw_features(rows.lo, rows.hi)
        if device.type == "cuda":
            need = sum(len(a) * np.dtype(t or a.dtype).itemsize for a, t in arrays.values())
            # Plus lo, n_events, order and row_user: int64 per row.
            need += sum(v.nbytes for v in (*profiles.values(), raw_features)) + 32 * len(rows)
            free = torch.cuda.mem_get_info(device)[0]
            if need > free / 2:
                raise UnsupportedCombinationError(
                    f"Token cache {cache.path} needs {need / 2**30:.1f} GiB on {device}, more "
                    f"than half of the {free / 2**30:.1f} GiB free. Set "
                    "`data.token_cache_on_device: false` to keep the cache on the host."
                )
        self.device = device
        self.arrays = {k: a.to_tensor(device, t) for k, (a, t) in arrays.items()}

        lo = cache.capped(rows, tokenizer.max_events)
        self.lo = torch.from_numpy(lo).to(device)
        self.n_events = torch.from_numpy(rows.hi - lo).to(device)
        self.order_on_device = torch.from_numpy(self.order).to(device)
        self.row_user = torch.from_numpy(row_user).to(device)
        self.profiles = {k: v.to(device) for k, v in profiles.items()}
        self.raw_features = raw_features.to(device)
        self.cutoff = ensure_utc(rows.cutoff).isoformat()

    def _collate(self, batch: slice) -> KeyValueTimeBatch:
        rows = self.order_on_device[batch]
        a = self.arrays
        max_events, max_tok, _ = self.pad_to

        lo, n_ev = self.lo[rows, None], self.n_events[rows, None]
        slot = torch.arange(max_events, device=self.device)
        event_ok = slot < n_ev
        event = torch.where(event_ok, lo + slot, 0)
        start = a["token_offsets"][event]
        count = a["token_offsets"][event + 1] - start
        tok_slot = torch.arange(max_tok, device=self.device)
        token_ok = event_ok[..., None] & (tok_slot < count[..., None])
        token = torch.where(token_ok, start[..., None] + tok_slot, 0)

        def tokens(name: str, fill: int) -> torch.Tensor:
            return torch.where(token_ok, a[name][token].long(), fill)

        def events(name: str) -> torch.Tensor:
            return torch.where(event_ok, a[name][event].long(), 0)

        id_us = a["id_us"][event]
        last = a["id_us"][lo + n_ev - 1]
        first = a["id_us"][lo]
        entity_ids = tuple(self.cache.user_ids(self.rows.user[self.order[batch]]))
        return KeyValueTimeBatch(
            key_ids=tokens("key_ids", PAD),
            value_ids=tokens("value_ids", PAD),
            positions=tokens("positions", 0),
            token_mask=token_ok.long(),
            event_mask=event_ok.long(),
            time_log=torch.where(event_ok, _time_encode(last - id_us), 0.0),
            elapsed_log=torch.where(event_ok, _time_encode(id_us - first), 0.0),
            events={name: events(name) for name in self.read_events},
            **{k: v[self.row_user[rows]] for k, v in self.profiles.items()},
            entity_ids=entity_ids,
            sequence_ids=tuple(f"{e}:{self.cutoff}" for e in entity_ids),
            fraud_labels=torch.zeros(len(entity_ids), dtype=torch.float32, device=self.device),
            numeric_features=self.raw_features[rows],
        )


__all__ = ["DeviceBatches"]
