# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token cache formats: how a recipe's token cache is checked, built, opened and read.

A tokenizer class names its cache's format in `token_cache_format` (default
`"columnar"`), by registered name or `"package.module:attribute"` (experimental).
Built-ins: `columnar`, any `ColumnarTokenizer`'s cache (`tfm.data.token_cache.columnar`),
and `flat_fields`, TabFormer's train split in `transaction_field_tokenizer` tokens
(`tfm.data.token_cache.flat`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Sized
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray

from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.registries import Registry
from tfm.contracts.tokenizers import SequenceTokenizer

if TYPE_CHECKING:
    from tfm.config import RecipeConfig

DEFAULT_TOKEN_CACHE_FORMAT = "columnar"


class PretrainRows(Sized, Protocol):
    """A cache's pretraining sequences, one per row."""

    @property
    def user(self) -> NDArray[np.int64]:
        """Each row's cached user."""
        ...


class TokenCacheFormat(Protocol):
    """A token cache format. `cache` is the object `open` returns."""

    trains_without_book: bool
    """Pretraining reads the cache alone (`prepare_pretraining`), cut at `train_cutoff`."""

    def config_problems(self, config: RecipeConfig) -> list[str]:
        """What this format cannot serve in `config`; checked when the recipe loads."""
        ...

    def build(
        self,
        config: RecipeConfig,
        tokenizer: SequenceTokenizer,
        *,
        workers: int,
        rows_per_block: int,
    ) -> dict[str, Any]:
        """Build the cache at `data.token_cache`; returns its manifest."""
        ...

    def profile(
        self,
        config: RecipeConfig,
        tokenizer: SequenceTokenizer,
        *,
        workers: int,
        rows_per_block: int,
        scratch: Path | None,
    ) -> dict[str, Any]:
        """Project `build` on this node."""
        ...

    def open(self, config: RecipeConfig, tokenizer: SequenceTokenizer) -> Any:
        """The cache at `data.token_cache`, checked against what `config` asks for."""
        ...

    def check_event_arrays(self, cache: Any, names: Sequence[str]) -> None:
        """Raise unless the cache holds every per-event array in `names`."""
        ...

    def run_tokenizer(
        self,
        cache: Any,
        tokenizer: SequenceTokenizer,
        config: RecipeConfig,
        *,
        users: Sequence[str],
        aliases: Mapping[str, str] | None,
    ) -> SequenceTokenizer | None:
        """The tokenizer a run uses for the sampled `users`, or None when the run fits
        `tokenizer` on its sample. `aliases` maps event ids that stand for cached ones."""
        ...

    def train_cutoff(self, cache: Any) -> datetime:
        """The train cutoff the cache was built at (`trains_without_book` formats)."""
        ...

    def pretrain_rows(
        self, cache: Any, config: RecipeConfig, *, train_cutoff: datetime
    ) -> PretrainRows: ...

    def batches(
        self,
        cache: Any,
        rows: PretrainRows,
        tokenizer: SequenceTokenizer,
        config: RecipeConfig,
        *,
        max_length: int,
        device: torch.device,
        event_arrays: Sequence[str],
    ) -> tuple[Sequence[Any], dict[str, Any]]:
        """Training minibatches of `rows`, each padded to `max_length` and carrying the
        per-event arrays `event_arrays`, the ones the model reads, and what the run
        records about how they were made. `device` is the run's device."""
        ...


_FORMATS: Registry[TokenCacheFormat] = Registry("token cache format")


def register_token_cache_format(
    name: str,
) -> Callable[[TokenCacheFormat], TokenCacheFormat]:
    def decorator(fmt: TokenCacheFormat) -> TokenCacheFormat:
        return _FORMATS.register(name, fmt)

    return decorator


def token_cache_format(name: str) -> TokenCacheFormat:
    """The format registered as `name`, or the object `"package.module:attribute"` names."""
    # Import both built-in formats so they register before lookup.
    import tfm.data.token_cache.columnar  # noqa: F401
    import tfm.data.token_cache.flat  # noqa: F401

    fmt = _FORMATS.get(name)
    if isinstance(fmt, type):
        raise UnsupportedCombinationError(
            f"token cache format {name!r} is a class; name an instance of it."
        )
    return cast(TokenCacheFormat, fmt)


def cache_format_of(tokenizer_cls: type) -> TokenCacheFormat:
    """The format of a tokenizer class's cache: its `token_cache_format`, else columnar."""
    return token_cache_format(
        getattr(tokenizer_cls, "token_cache_format", DEFAULT_TOKEN_CACHE_FORMAT)
    )


def cache_path(config: RecipeConfig) -> Path:
    if config.data.token_cache is None:
        raise UnsupportedCombinationError("data.token_cache is not set in this config.")
    return Path(config.data.token_cache).expanduser()


__all__ = [
    "DEFAULT_TOKEN_CACHE_FORMAT",
    "PretrainRows",
    "TokenCacheFormat",
    "cache_format_of",
    "cache_path",
    "register_token_cache_format",
    "token_cache_format",
]
