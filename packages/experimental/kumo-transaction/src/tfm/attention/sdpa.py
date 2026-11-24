# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused scaled-dot-product attention through `torch.nn.functional`.

Same semantics as `EagerAttention`, but PyTorch chooses the kernel: on CUDA it
can dispatch to FlashAttention or the memory-efficient backend, which never
materializes the `[B, H, T, T]` score matrix. That matters for the PRAGMA stack,
whose history encoder runs over long event sequences.

`EagerAttention` stays the correctness baseline; the equivalence of the two is
asserted in the contract tests, as the performance rules require.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tfm.layers.masks import attention_bias


@dataclass(frozen=True)
class SdpaAttention:
    """Composition token: use PyTorch's fused SDPA kernel.

    Not part of the core export surface. Behavior must match `EagerAttention`
    to floating-point tolerance for any mask this kit builds.
    """

    name: str = "sdpa"
    support_tier: str = "experimental"

    def capabilities(self) -> dict[str, bool]:
        return {"causal": True, "bidirectional": True}

    def build(self) -> nn.Module:
        return SdpaAttentionKernel()


class SdpaAttentionKernel(nn.Module):
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
        visible = attention_bias(attention_mask, causal=causal, segment_ids=segment_ids)
        # A fully masked query row would make SDPA's softmax produce NaN, while
        # the eager kernel returns zeros. Let such rows attend to themselves and
        # zero them afterwards, so both kernels agree.
        dead = ~visible.any(dim=-1, keepdim=True)
        eye = torch.eye(query.size(-2), dtype=torch.bool, device=query.device)
        visible = visible | (dead & eye)
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=visible)
        return out.masked_fill(dead, 0.0)
