# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name registry for sequence tokenizers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.tokenizers import SequenceTokenizer

_TOKENIZERS: dict[str, type] = {}


def register_tokenizer(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        _TOKENIZERS[name] = cls
        return cls

    return decorator


def advertised_tokenizers() -> dict[str, type]:
    return dict(_TOKENIZERS)


def build_tokenizer(name: str, **kwargs: Any) -> SequenceTokenizer:
    if name not in _TOKENIZERS:
        _try_register_extra_tokenizer(name)
    if name not in _TOKENIZERS:
        known = ", ".join(sorted(_TOKENIZERS)) or "<none>"
        raise UnsupportedCombinationError(
            f"Unknown tokenizer {name!r}. Registered tokenizers: {known}."
        )
    return _TOKENIZERS[name](**kwargs)


def _try_register_extra_tokenizer(name: str) -> None:
    if name == "rapids_field_tokenizer":
        from tfm.extras.rapids import register as register_rapids

        register_rapids()
