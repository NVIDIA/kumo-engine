# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")


@dataclass(frozen=True)
class MapLikeCase(Generic[_K, _V]):
    key: _K
    value: _V
    key_type: type[_K]
    value_type: type[_V]


CaseBuilder = Callable[[], MapLikeCase[_K, _V]]
