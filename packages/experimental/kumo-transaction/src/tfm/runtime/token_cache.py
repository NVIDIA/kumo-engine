# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A recipe's token cache, through its tokenizer's format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tfm.config import RecipeConfig
from tfm.contracts.tokenizers import SequenceTokenizer
from tfm.data.token_cache.formats import cache_format_of


def build_recipe_cache(
    config: RecipeConfig,
    tokenizer: SequenceTokenizer,
    *,
    workers: int,
    rows_per_block: int = 1_000_000,
) -> dict[str, Any]:
    return cache_format_of(type(tokenizer)).build(
        config, tokenizer, workers=workers, rows_per_block=rows_per_block
    )


def profile_recipe_cache(
    config: RecipeConfig,
    tokenizer: SequenceTokenizer,
    *,
    workers: int,
    rows_per_block: int = 1_000_000,
    scratch: Path | None = None,
) -> dict[str, Any]:
    """Project the build `build_recipe_cache` would run."""
    return cache_format_of(type(tokenizer)).profile(
        config, tokenizer, workers=workers, rows_per_block=rows_per_block, scratch=scratch
    )


def open_recipe_cache(config: RecipeConfig, tokenizer: SequenceTokenizer) -> Any:
    """The configured cache, checked against the book, sample or fit, and tokenizer
    this config asks for."""
    return cache_format_of(type(tokenizer)).open(config, tokenizer)


__all__ = ["build_recipe_cache", "open_recipe_cache", "profile_recipe_cache"]
