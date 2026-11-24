# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager scaled-dot-product attention (no FlashAttention)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tfm.layers.masks import attention_bias


@dataclass(frozen=True)
class EagerAttention:
    """Public composition token: use the PyTorch eager attention kernel."""

    name: str = "eager"

    def capabilities(self) -> dict[str, bool]:
        return {"causal": True, "bidirectional": True}

    def build(self) -> nn.Module:
        return EagerAttentionKernel()


class EagerAttentionKernel(nn.Module):
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        *,
        causal: bool,
        segment_ids: Tensor | None = None,
    ) -> Tensor:
        """query/key/value: [B, H, T, D]."""
        scale = 1.0 / math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        visible = attention_bias(attention_mask, causal=causal, segment_ids=segment_ids)
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~visible, 0.0)
        return torch.matmul(weights, value)
