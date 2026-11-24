# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mean over valid positions (experimental)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class MeanPool:
    """Average every valid state. Not a core export.

    Under a causal mask, position `t` has seen positions `0..t`, so the mean
    still depends only on the history up to the cutoff.
    """

    name: str = "mean"
    support_tier: str = "experimental"


class MeanPoolModule(nn.Module):
    def forward(self, states: Tensor, attention_mask: Tensor) -> Tensor:
        weights = attention_mask.to(states.dtype).unsqueeze(-1)
        return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
