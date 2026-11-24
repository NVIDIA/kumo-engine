# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A recipe's token cache: what it is built from, building it, and opening it (ADR 0016)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from tfm.config import RecipeConfig
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.tokenizers import SequenceTokenizer
from tfm.data.adapters.paths import entity_cap, resolve_data_path, sample_seed
from tfm.data.adapters.tabformer import precomputed_sample, tabformer_label_delay
from tfm.data.splits.temporal import parse_duration, parse_timestamp
from tfm.data.token_cache import CacheRequest, FitSpec, TokenCache, build_token_cache
from tfm.data.token_cache.flat import FlatCacheRequest, FlatTokenCache, build_flat_cache
from tfm.features.tokenizers.financial import TransactionFieldTokenizer
from tfm.features.tokenizers.key_value_time import KeyValueTimeTokenizer


def cache_path(config: RecipeConfig) -> Path:
    if config.data.token_cache is None:
        raise UnsupportedCombinationError("data.token_cache is not set in this config.")
    return Path(config.data.token_cache).expanduser()


def cache_request(config: RecipeConfig, tokenizer: SequenceTokenizer) -> CacheRequest:
    """Every user of the configured book, fit as `prepare_data` fits its train side.

    `tokenizer` is the unfitted `tokenizer_for(config)`; only its settings are read.
    """
    if type(tokenizer) is not KeyValueTimeTokenizer:
        raise UnsupportedCombinationError(
            f"A token cache holds key_value_time tokens; {tokenizer.name!r} encodes differently."
        )
    settings = cast(KeyValueTimeTokenizer, tokenizer).config
    return CacheRequest(
        root=resolve_data_path(config.data),
        users=None,
        fit=FitSpec(
            train_end=parse_timestamp(config.data.temporal_split.train_end),
            observation_window=parse_duration(config.data.observation_window),
            max_events=config.data.sequence_length,
        ),
        tokenizer=settings,
        population={"users": "all"},
    )


def flat_cache_request(config: RecipeConfig, tokenizer: SequenceTokenizer) -> FlatCacheRequest:
    """The train split of the TabFormer sample `prepare_data` loads, in this tokenizer's fields."""
    if type(tokenizer) is not TransactionFieldTokenizer:
        raise UnsupportedCombinationError(
            "A flat-field token cache holds transaction_field_tokenizer tokens; "
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
    )


def _is_flat(tokenizer: SequenceTokenizer) -> bool:
    return isinstance(tokenizer, TransactionFieldTokenizer)


def build_recipe_cache(
    config: RecipeConfig,
    tokenizer: SequenceTokenizer,
    *,
    workers: int,
    rows_per_block: int = 1_000_000,
) -> dict[str, Any]:
    if _is_flat(tokenizer):
        return build_flat_cache(
            flat_cache_request(config, tokenizer),
            cache_path(config),
            workers=workers,
            rows_per_block=rows_per_block,
        )
    return build_token_cache(
        cache_request(config, tokenizer),
        cache_path(config),
        workers=workers,
        rows_per_block=rows_per_block,
    )


def open_recipe_cache(
    config: RecipeConfig, tokenizer: SequenceTokenizer
) -> TokenCache | FlatTokenCache:
    """The configured cache, checked against the book, sample or fit, and tokenizer
    this config asks for."""
    if _is_flat(tokenizer):
        flat = FlatTokenCache(cache_path(config))
        flat.check_identity(flat_cache_request(config, tokenizer).identity())
        return flat
    cache = TokenCache(cache_path(config))
    cache.check_identity(cache_request(config, tokenizer).identity())
    return cache


__all__ = [
    "build_recipe_cache",
    "cache_path",
    "cache_request",
    "flat_cache_request",
    "open_recipe_cache",
]
