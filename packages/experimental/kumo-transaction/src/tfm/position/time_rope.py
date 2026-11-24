# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Continuous-time rotary embeddings (experimental)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@dataclass(frozen=True)
class TimeRoPE:
    """Public composition token for continuous-time RoPE. Not a core export."""

    name: str = "time_rope"
    base: float = 10000.0
    support_tier: str = "experimental"


class TimeRoPEModule(nn.Module):
    """Rotate query/key by a real-valued log-seconds position."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("TimeRoPE head_dim must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, query: Tensor, key: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """query/key: [B, H, T, Dh]; positions: [B, T] float seconds-transform."""
        freqs = torch.einsum("bt,d->btd", positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        return query, key
