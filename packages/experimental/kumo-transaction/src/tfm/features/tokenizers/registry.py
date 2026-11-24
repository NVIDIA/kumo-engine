# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name registry for sequence tokenizers."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.registries import Registry
from tfm.contracts.tokenizers import SequenceTokenizer

if TYPE_CHECKING:
    from tfm.config import DataConfig, FeaturesConfig, RecipeConfig

_TOKENIZERS: Registry[type] = Registry("tokenizer")

FEATURE_KEYS = ("fields", "categorical_keys", "hashed_keys", "hash_buckets", "exclude_keys")
"""The `features` keys a tokenizer class may read through `from_config`."""


@runtime_checkable
class ConfiguredTokenizer(Protocol):
    """A tokenizer class that builds itself from the recipe's `features` and `data`.

    `feature_keys` lists the `FEATURE_KEYS` it reads; a run refuses any other one that
    is set. A class without `from_config` is built from `features.tokenizer_options`.
    """

    feature_keys: ClassVar[tuple[str, ...]]

    @classmethod
    def from_config(cls, features: FeaturesConfig, data: DataConfig) -> SequenceTokenizer: ...


@runtime_checkable
class FixedWidthTokenizer(Protocol):
    """A tokenizer class whose every event is `event_width(features)` tokens, its
    boundary token included; a run checks its pretraining windows fit their rows."""

    @classmethod
    def event_width(cls, features: FeaturesConfig) -> int: ...


def register_tokenizer[TokenizerT: type](name: str) -> Callable[[TokenizerT], TokenizerT]:
    def decorator(cls: TokenizerT) -> TokenizerT:
        _TOKENIZERS.register(name, cls)
        return cls

    return decorator


def advertised_tokenizers() -> dict[str, type]:
    return _TOKENIZERS.entries()


def tokenizer_class(name: str) -> type:
    """The class registered as `name`, or the one `"package.module:Class"` names."""
    if ":" not in name and name not in _TOKENIZERS:
        _try_register_extra_tokenizer(name)
    cls = _TOKENIZERS.get(name)
    if not isinstance(cls, type):
        raise UnsupportedCombinationError(
            f"features.tokenizer {name!r} is a {type(cls).__name__}, not a class."
        )
    return cls


def build_tokenizer(name: str, **kwargs: Any) -> SequenceTokenizer:
    return tokenizer_class(name)(**kwargs)


def _try_register_extra_tokenizer(name: str) -> None:
    if name == "rapids_field_tokenizer":
        from tfm.extras.rapids import register as register_rapids

        register_rapids()


def tokenizer_for(config: RecipeConfig) -> SequenceTokenizer:
    """The tokenizer `features.tokenizer` names: built from `features.tokenizer_options`
    when set, else by the class's `from_config` (`ConfiguredTokenizer`), else with no
    arguments. A `features` key the class does not read is refused."""
    features = config.features
    if features.tokenizer_options is not None:
        return _tokenizer_from_options(features, features.tokenizer_options)
    cls = tokenizer_class(features.tokenizer)
    reads = cls.feature_keys if isinstance(cls, ConfiguredTokenizer) else ()
    for key in FEATURE_KEYS:
        if _is_set(features, key) and key not in reads:
            raise UnsupportedCombinationError(
                f"features.{key} does not apply to tokenizer {features.tokenizer!r}, which "
                f"reads {list(reads)}."
            )
    if isinstance(cls, ConfiguredTokenizer):
        return cls.from_config(features, config.data)
    tokenizer: SequenceTokenizer = cls()
    return tokenizer


def _is_set(features: FeaturesConfig, key: str) -> bool:
    """`features.<key>` differs from its default (an empty list set on purpose counts)."""
    default = type(features).model_fields[key].get_default(call_default_factory=True)
    return getattr(features, key) != default


def _tokenizer_from_options(features: FeaturesConfig, options: dict[str, Any]) -> SequenceTokenizer:
    if set_keys := [k for k in FEATURE_KEYS if _is_set(features, k)]:
        raise UnsupportedCombinationError(
            f"features.tokenizer_options and features.{', '.join(set_keys)} are both set; "
            "pass every setting through tokenizer_options."
        )
    cls = tokenizer_class(features.tokenizer)
    try:
        inspect.signature(cls).bind(**options)
    except TypeError as error:
        raise UnsupportedCombinationError(
            f"features.tokenizer_options do not fit {cls.__qualname__}{inspect.signature(cls)}: "
            f"{error}."
        ) from error
    tokenizer: SequenceTokenizer = cls(**options)
    return tokenizer
