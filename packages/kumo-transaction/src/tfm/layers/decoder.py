# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal transformer decoder used by the Flat composition."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from tfm.attention.eager import EagerAttention
from tfm.contracts.protocols import AttentionSpec


@dataclass(frozen=True)
class Decoder:
    depth: int = 2
    mask: str = "causal"
    hidden_size: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    vocab_size: int = 4096
    max_seq_len: int = 256
    ffn_size: int | None = None

    def __post_init__(self) -> None:
        if self.mask != "causal":
            from tfm.contracts.errors import IncompatibleCompositionError

            raise IncompatibleCompositionError(
                f"Decoder only supports mask='causal', got {self.mask!r}."
            )
        if self.hidden_size % self.num_heads != 0:
            from tfm.contracts.errors import IncompatibleCompositionError

            raise IncompatibleCompositionError("hidden_size must be divisible by num_heads.")


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        ffn_size: int,
        attention: nn.Module,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm1 = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn = attention
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
        *,
        causal: bool,
        segment_ids: Tensor | None = None,
    ) -> Tensor:
        batch, seq, dim = hidden.shape
        normed = self.norm1(hidden)
        qkv = self.qkv(normed).view(batch, seq, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn_out = self.attn(
            query, key, value, attention_mask, causal=causal, segment_ids=segment_ids
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, dim)
        hidden = hidden + self.drop(self.out(attn_out))
        hidden = hidden + self.ff(self.norm2(hidden))
        return hidden


class DecoderModule(nn.Module):
    def __init__(self, spec: Decoder, attention: AttentionSpec | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.attention_spec: AttentionSpec = (
            attention if attention is not None else EagerAttention()
        )
        ffn = spec.ffn_size if spec.ffn_size is not None else 4 * spec.hidden_size
        self.embed = nn.Embedding(spec.vocab_size, spec.hidden_size)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    spec.hidden_size,
                    spec.num_heads,
                    spec.dropout,
                    ffn,
                    self.attention_spec.build(),
                )
                for _ in range(spec.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(spec.hidden_size)
        self.lm_head = nn.Linear(spec.hidden_size, spec.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        causal: bool = True,
        segment_ids: Tensor | None = None,
    ) -> Tensor:
        hidden = self.embed(input_ids)
        for block in self.blocks:
            hidden = block(hidden, attention_mask, causal=causal, segment_ids=segment_ids)
        return self.final_norm(hidden)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.lm_head(hidden)
