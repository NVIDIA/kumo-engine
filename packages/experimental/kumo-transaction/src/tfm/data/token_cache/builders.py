# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch builders: what turns a token cache's rows into training batches.

`data.batch_builder` selects one by registered name or `"package.module:attribute"`
(experimental until the packed batch). The default, `token_cache`, collates on the
GPU from arrays held there when the run places the cache on the device
(`data.token_cache_on_device`), else on the host.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

import torch

from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.kvt import KeyValueTimeBatch
from tfm.contracts.registries import Registry
from tfm.data.token_cache.device import DeviceBatches
from tfm.data.token_cache.store import CachedBatches, CachedTokenizer, CacheRows, TokenCache

DEFAULT_BATCH_BUILDER = "token_cache"


class BatchBuilder(Protocol):
    """Minibatches of `rows`, in a `shuffle_seed` order (none: row order), each padded to
    `max_length` tokens per event, collated on `device` (the run's placement: a CUDA
    device, or the CPU for the host). Each batch carries the per-event arrays
    `event_arrays`, the ones the model reads, and need carry no other."""

    def __call__(
        self,
        cache: TokenCache,
        rows: CacheRows,
        tokenizer: CachedTokenizer,
        *,
        batch_size: int,
        max_length: int,
        shuffle_seed: int | None,
        device: torch.device,
        event_arrays: Sequence[str],
    ) -> Sequence[KeyValueTimeBatch]: ...


_BATCH_BUILDERS: Registry[BatchBuilder] = Registry("batch builder")


def register_batch_builder(name: str) -> Callable[[BatchBuilder], BatchBuilder]:
    def decorator(builder: BatchBuilder) -> BatchBuilder:
        return _BATCH_BUILDERS.register(name, builder)

    return decorator


def batch_builder(name: str) -> BatchBuilder:
    """The builder registered as `name`, or the callable `"package.module:attribute"` names."""
    builder = _BATCH_BUILDERS.get(name)
    if not callable(builder):
        raise UnsupportedCombinationError(
            f"data.batch_builder {name!r} is a {type(builder).__name__}, not a callable."
        )
    return cast(BatchBuilder, builder)


@register_batch_builder(DEFAULT_BATCH_BUILDER)
def token_cache_batches(
    cache: TokenCache,
    rows: CacheRows,
    tokenizer: CachedTokenizer,
    *,
    batch_size: int,
    max_length: int,
    shuffle_seed: int | None,
    device: torch.device,
    event_arrays: Sequence[str],
) -> Sequence[KeyValueTimeBatch]:
    """`DeviceBatches` on a CUDA `device`, else `CachedBatches`."""
    if device.type == "cuda":
        return DeviceBatches(
            cache,
            rows,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle_seed=shuffle_seed,
            device=device,
            event_arrays=event_arrays,
        )
    return CachedBatches(
        cache,
        rows,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle_seed=shuffle_seed,
        event_arrays=event_arrays,
    )


__all__ = [
    "DEFAULT_BATCH_BUILDER",
    "BatchBuilder",
    "batch_builder",
    "register_batch_builder",
    "token_cache_batches",
]
