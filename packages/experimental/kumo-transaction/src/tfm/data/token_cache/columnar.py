# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `columnar` token cache format: a recipe's book through any `ColumnarTokenizer`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.tokenizers import ColumnarTokenizer, SequenceTokenizer
from tfm.data.adapters.registry import columnar_source
from tfm.data.splits.temporal import parse_duration, parse_timestamp
from tfm.data.token_cache.build import CacheRequest, FitSpec, build_token_cache
from tfm.data.token_cache.builders import DEFAULT_BATCH_BUILDER, batch_builder
from tfm.data.token_cache.formats import (
    DEFAULT_TOKEN_CACHE_FORMAT,
    PretrainRows,
    cache_path,
    register_token_cache_format,
)
from tfm.data.token_cache.profile import profile_token_cache
from tfm.data.token_cache.store import CachedTokenizer, CacheRows, TokenCache

if TYPE_CHECKING:
    from tfm.config import RecipeConfig


def cache_request(config: RecipeConfig, tokenizer: SequenceTokenizer) -> CacheRequest:
    """Every user of the configured book, fit as `prepare_data` fits its train side.

    `tokenizer` is the unfitted `tokenizer_for(config)`.
    """
    if not isinstance(tokenizer, ColumnarTokenizer):
        raise UnsupportedCombinationError(
            f"{type(tokenizer).__qualname__} cannot build a token cache: it does not "
            "implement `tfm.contracts.tokenizers.ColumnarTokenizer`."
        )
    return CacheRequest(
        source=columnar_source(config.data),
        users=None,
        fit=FitSpec(
            train_end=parse_timestamp(config.data.temporal_split.train_end),
            observation_window=parse_duration(config.data.observation_window),
            max_events=config.data.sequence_length,
        ),
        tokenizer=cast(ColumnarTokenizer[Any], tokenizer),
        population={"users": "all"},
        amount_field=config.data.amount_field,
    )


class ColumnarCacheFormat:
    """Every user of the book, fit on its train side; read through `data.batch_builder`."""

    trains_without_book = False

    def config_problems(self, config: RecipeConfig) -> list[str]:
        data = config.data
        problems: list[str] = []
        if data.split_protocol != "timestamp":
            problems.append(
                "data.token_cache fits on a timestamp split; split_protocol is precomputed"
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
        return build_token_cache(
            cache_request(config, tokenizer),
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
        return profile_token_cache(
            cache_request(config, tokenizer),
            workers=workers,
            rows_per_block=rows_per_block,
            scratch=scratch,
        )

    def open(self, config: RecipeConfig, tokenizer: SequenceTokenizer) -> TokenCache:
        cache = TokenCache(cache_path(config))
        cache.check_identity(cache_request(config, tokenizer).identity())
        return cache

    def check_event_arrays(self, cache: TokenCache, names: Sequence[str]) -> None:
        cache.check_event_arrays(names)

    def run_tokenizer(
        self,
        cache: TokenCache,
        tokenizer: SequenceTokenizer,
        config: RecipeConfig,
        *,
        users: Sequence[str],
        aliases: Mapping[str, str] | None,
    ) -> CachedTokenizer:
        """The cache's tokenizer; each row keeps its last `data.sequence_length` events.
        Every sampled user must be cached."""
        outside = [u for u, i in zip(users, cache.user_index(users).tolist(), strict=True) if i < 0]
        if outside:
            raise SchemaError(
                f"{len(outside)} sampled users are not in token cache {cache.path} "
                f"(first: {outside[0]!r}); it must hold every probe user."
            )
        cached = cache.tokenizer(max_events=config.data.sequence_length)
        if aliases:
            cached.add_aliases(aliases)
        return cached

    def train_cutoff(self, cache: TokenCache) -> datetime:
        raise UnsupportedCombinationError(
            "A columnar token cache is read with the book (prepare_data), which loads "
            "profiles and probes; it has no pretraining-only path."
        )

    def pretrain_rows(
        self, cache: TokenCache, config: RecipeConfig, *, train_cutoff: datetime
    ) -> CacheRows:
        """Every cached user's train-side history, or its windows (`data.pretrain_windows`)."""
        lookback = parse_duration(config.data.observation_window)
        windows = config.data.pretrain_windows
        if windows is None:
            return cache.train_rows(
                train_end=train_cutoff,
                lookback=lookback,
                max_events=config.data.sequence_length,
            )
        return cache.train_windows(
            train_end=train_cutoff, lookback=lookback, window_events=windows.events
        )

    def batches(
        self,
        cache: TokenCache,
        rows: PretrainRows,
        tokenizer: SequenceTokenizer,
        config: RecipeConfig,
        *,
        max_length: int,
        device: torch.device,
        event_arrays: Sequence[str],
    ) -> tuple[Sequence[Any], dict[str, Any]]:
        """`data.batch_builder`'s batches, collated on `device` when
        `data.token_cache_on_device` and `device` is CUDA, else on the host."""
        on_device = config.data.token_cache_on_device and device.type == "cuda"
        name = config.data.batch_builder
        batches = batch_builder(name)(
            cache,
            cast(CacheRows, rows),
            cast(CachedTokenizer, tokenizer),
            batch_size=config.training.batch_size,
            max_length=max_length,
            shuffle_seed=config.seed if config.training.shuffle else None,
            device=device if on_device else torch.device("cpu"),
            event_arrays=event_arrays,
        )
        return batches, {
            "token_cache_on_device": on_device,
            "batch_builder": name,
            "batch_builder_default": name == DEFAULT_BATCH_BUILDER,
        }


COLUMNAR = register_token_cache_format(DEFAULT_TOKEN_CACHE_FORMAT)(ColumnarCacheFormat())

__all__ = ["COLUMNAR", "ColumnarCacheFormat", "cache_request"]
