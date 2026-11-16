# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small configuration helpers shared by native and legacy training."""

from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T", bound="CastMixin")


class CastMixin:
    @classmethod
    def cast(cls: type[T], *args: Any, **kwargs: Any) -> T:
        if len(args) == 1 and not kwargs:
            value = args[0]
            if value is None:
                return value
            if isinstance(value, CastMixin):
                return value
            if isinstance(value, (tuple, list)):
                return cls(*value)
            if isinstance(value, dict):
                return cls(**value)
        return cls(*args, **kwargs)


__all__ = ["CastMixin"]
