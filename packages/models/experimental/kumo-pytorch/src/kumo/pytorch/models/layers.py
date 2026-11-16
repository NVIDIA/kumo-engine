# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal pre-normalized Transformer layers with optional time-based RoPE."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_positions(position_ids: torch.Tensor, width: int) -> torch.Tensor:
    """Return fixed sine/cosine features for positions within one field value."""

    if width % 2:
        raise ValueError("sinusoidal width must be even")
    half = width // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=position_ids.device, dtype=torch.float32)
        / max(1, half - 1)
    )
    angles = position_ids.to(torch.float32).unsqueeze(-1) * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


def apply_temporal_rope(
    query: torch.Tensor, key: torch.Tensor, coordinates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate query/key pairs using an arbitrary scalar coordinate per token."""

    head_width = query.shape[-1]
    if head_width % 2:
        raise ValueError("attention head width must be even for rotary embeddings")
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(0, head_width, 2, device=query.device, dtype=torch.float32)
        / head_width
    )
    angles = coordinates.to(torch.float32)[:, None, :, None] * frequencies[None, None, None]
    cosine = angles.cos().to(query.dtype)
    sine = angles.sin().to(query.dtype)

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
        return rotated.flatten(-2)

    return rotate(query), rotate(key)


class SelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def forward(
        self,
        inputs: torch.Tensor,
        valid_tokens: torch.Tensor,
        temporal_coordinates: torch.Tensor | None = None,
        *,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, length, _width = inputs.shape
        qkv = self.qkv(inputs).view(batch, length, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if temporal_coordinates is not None:
            query, key = apply_temporal_rope(query, key, temporal_coordinates)
        allowed_keys = valid_tokens[:, None, None, :]
        if causal:
            allowed_keys = (
                allowed_keys
                & torch.ones((length, length), dtype=torch.bool, device=inputs.device).tril()[
                    None, None
                ]
            )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed_keys,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.width)
        return self.output(attended) * valid_tokens.unsqueeze(-1)

    def forward_packed(
        self,
        inputs: torch.Tensor,
        offsets: torch.Tensor,
        *,
        min_length: int,
        max_length: int,
    ) -> torch.Tensor:
        """Attend within variable-length sequences stored in one compact buffer."""

        total_tokens = inputs.shape[0]
        qkv = self.qkv(inputs).view(total_tokens, 3, self.heads, self.head_width)
        query_values, key_values, value_values = qkv.unbind(dim=1)

        def as_nested(values: torch.Tensor) -> torch.Tensor:
            return torch.nested.nested_tensor_from_jagged(
                values.contiguous(),
                offsets,
                min_seqlen=min_length,
                max_seqlen=max_length,
            ).transpose(1, 2)

        query, key, value = (
            as_nested(values) for values in (query_values, key_values, value_values)
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended_values = attended.transpose(1, 2).values().flatten(-2)
        return self.output(attended_values)


class TransformerBlock(nn.Module):
    def __init__(self, width: int, ffn_width: int, heads: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = SelfAttention(width, heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_width, width),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        valid_tokens: torch.Tensor,
        temporal_coordinates: torch.Tensor | None = None,
        *,
        causal: bool = False,
    ) -> torch.Tensor:
        inputs = inputs + self.attention_dropout(
            self.attention(
                self.attention_norm(inputs),
                valid_tokens,
                temporal_coordinates,
                causal=causal,
            )
        )
        inputs = inputs + self.ffn(self.ffn_norm(inputs))
        return inputs * valid_tokens.unsqueeze(-1)

    def forward_packed(
        self,
        inputs: torch.Tensor,
        offsets: torch.Tensor,
        *,
        min_length: int,
        max_length: int,
    ) -> torch.Tensor:
        """Apply this block to a compact jagged token buffer."""

        inputs = inputs + self.attention_dropout(
            self.attention.forward_packed(
                self.attention_norm(inputs),
                offsets,
                min_length=min_length,
                max_length=max_length,
            )
        )
        return inputs + self.ffn(self.ffn_norm(inputs))


class TransformerStack(nn.Module):
    def __init__(self, layers: int, width: int, ffn_width: int, heads: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(width, ffn_width, heads, dropout) for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(
        self,
        inputs: torch.Tensor,
        valid_tokens: torch.Tensor,
        temporal_coordinates: torch.Tensor | None = None,
        *,
        causal: bool = False,
    ) -> torch.Tensor:
        for block in self.blocks:
            inputs = block(inputs, valid_tokens, temporal_coordinates, causal=causal)
        return self.output_norm(inputs) * valid_tokens.unsqueeze(-1)

    def forward_packed(
        self,
        inputs: torch.Tensor,
        offsets: torch.Tensor,
        *,
        min_length: int,
        max_length: int,
    ) -> torch.Tensor:
        """Encode variable-length sequences without materializing padding."""

        for block in self.blocks:
            inputs = block.forward_packed(
                inputs,
                offsets,
                min_length=min_length,
                max_length=max_length,
            )
        return self.output_norm(inputs)
