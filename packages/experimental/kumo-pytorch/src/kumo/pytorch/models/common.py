# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact neural-network pieces reused where the papers use the same idea."""

from __future__ import annotations

import torch
from torch import nn


class LowRankLinear(nn.Module):
    """A frozen linear layer with a trainable LoRA update."""

    def __init__(self, base: nn.Linear, rank: int, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.up(self.down(self.dropout(inputs)))
