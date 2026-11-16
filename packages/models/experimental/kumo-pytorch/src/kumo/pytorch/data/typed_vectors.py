# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model containers for fixed-schema numerical and categorical vectors."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch


@dataclass(frozen=True)
class AttributeSchema:
    """Ordered fields and per-field categorical vocabularies."""

    numerical: tuple[str, ...]
    categorical: tuple[str, ...]
    categories: dict[str, tuple[Any, ...]]
    high_cardinality: frozenset[str] = frozenset()

    def cardinality(self, name: str) -> int:
        return len(self.categories[name])


@dataclass
class TypedVectorBatch:
    """One static vector and a padded sequence of dynamic vectors per entity."""

    entity_ids: torch.Tensor
    static_numerical: torch.Tensor
    static_numerical_present: torch.Tensor
    static_categorical: torch.Tensor
    dynamic_numerical: torch.Tensor
    dynamic_numerical_present: torch.Tensor
    dynamic_categorical: torch.Tensor
    dynamic_categorical_present: torch.Tensor
    signal_numerical: torch.Tensor
    signal_numerical_present: torch.Tensor
    signal_categorical: torch.Tensor
    signal_categorical_present: torch.Tensor
    event_mask: torch.Tensor

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> TypedVectorBatch:
        return TypedVectorBatch(
            **{
                field.name: getattr(self, field.name).to(device, non_blocking=non_blocking)
                for field in fields(self)
            }
        )

    @property
    def batch_size(self) -> int:
        return int(self.entity_ids.numel())

    @property
    def token_count(self) -> int:
        return int(self.event_mask.sum().item())
