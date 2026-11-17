# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers shared by the declarations in this package."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import torch

from kumo.pytorch.contract._frozen import FrozenMapping


def _mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return FrozenMapping(dict(value))


def _probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between zero and one")


def _move(value: Any, device: torch.device | str, non_blocking: bool) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {name: _move(item, device, non_blocking) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [_move(item, device, non_blocking) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                item.name: _move(getattr(value, item.name), device, non_blocking)
                for item in fields(value)
            }
        )
    return value
