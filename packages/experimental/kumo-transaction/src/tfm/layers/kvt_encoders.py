# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Building blocks shared by the key-value-time models: PRAGMA-like and Blueprint v2.

Both encode each event's tokens into an `[EVT]` summary and the profile's
tokens into a `[USR]` summary with the same modules; they differ only in the
history model above those summaries.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor, nn

from tfm.contracts.protocols import AttentionSpec
from tfm.layers.varlen import compress_rows, scatter_rows
from tfm.position.time_rope import TimeRoPEModule


class KVTEncoderSpec(Protocol):
    """The granularity fields the shared encoders read."""

    @property
    def hidden_size(self) -> int: ...
    @property
    def num_heads(self) -> int: ...
    @property
    def dropout(self) -> float: ...
    @property
    def rope_base(self) -> float: ...
    @property
    def event_depth(self) -> int: ...
    @property
    def profile_depth(self) -> int: ...
    @property
    def varlen(self) -> bool: ...


def sinusoidal_position(positions: Tensor, dim: int, max_position: int) -> Tensor:
    pos = positions.clamp(min=0, max=max_position - 1).float()
    half = dim // 2
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=positions.device, dtype=torch.float32) / half
    )
    angles = pos.unsqueeze(-1) * freq
    out = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if dim % 2 == 1:
        out = nn.functional.pad(out, (0, 1))
    return out


class CalendarEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, hour: Tensor, dow: Tensor, dom: Tensor) -> Tensor:
        h = 2.0 * math.pi * hour.float() / 24.0
        w = 2.0 * math.pi * dow.float() / 7.0
        d = 2.0 * math.pi * (dom.float() - 1.0) / 31.0
        feats = torch.stack([h.sin(), h.cos(), w.sin(), w.cos(), d.sin(), d.cos()], dim=-1)
        return self.net(feats)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        *,
        use_rope: bool,
        rope_base: float,
        attention: nn.Module,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm1 = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn = attention
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.rope = TimeRoPEModule(self.head_dim, rope_base) if use_rope else None

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
        *,
        rope_positions: Tensor | None = None,
    ) -> Tensor:
        batch, seq, dim = hidden.shape
        normed = self.norm1(hidden)
        qkv = self.qkv(normed).view(batch, seq, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        if self.rope is not None and rope_positions is not None:
            query, key = self.rope(query, key, rope_positions)
        attn_out = self.attn(query, key, value, attention_mask, causal=self.causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, dim)
        hidden = hidden + self.drop(self.out(attn_out))
        return hidden + self.ff(self.norm2(hidden))


def encoder_blocks(
    spec: KVTEncoderSpec,
    attention: AttentionSpec,
    depth: int,
    *,
    use_rope: bool,
    causal: bool,
) -> nn.ModuleList:
    return nn.ModuleList(
        [
            EncoderBlock(
                spec.hidden_size,
                spec.num_heads,
                spec.dropout,
                use_rope=use_rope,
                rope_base=spec.rope_base,
                attention=attention.build(),
                causal=causal,
            )
            for _ in range(depth)
        ]
    )


def _summary(learned: nn.Parameter | None, given: Tensor | None, name: str) -> Tensor:
    if given is not None:
        return given
    if learned is None:
        raise ValueError(f"This encoder has no learned {name} vector; pass `summary`.")
    return learned


class KVTEventEncoder(nn.Module):
    """`EventEncoder`: bidirectional within each event, `[EVT]` slot as the summary.

    The `[EVT]` input is a learned vector, or, with `learned_summary=False`, the
    `summary` passed to `forward` (PRAGMA ties it to an embedding row).
    """

    def __init__(
        self, spec: KVTEncoderSpec, attention: AttentionSpec, *, learned_summary: bool = True
    ) -> None:
        super().__init__()
        self.varlen = spec.varlen
        self.blocks = encoder_blocks(
            spec, attention, spec.event_depth, use_rope=False, causal=False
        )
        self.norm = nn.LayerNorm(spec.hidden_size)
        self.evt = nn.Parameter(torch.randn(spec.hidden_size) * 0.02) if learned_summary else None
        self.last_padding_fraction: float = 0.0
        """Share of the event grid skipped by the last varlen forward. 0 when padded."""

    def _run(self, hidden: Tensor, mask: Tensor) -> Tensor:
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.norm(hidden)

    def forward(
        self,
        tokens: Tensor,
        token_mask: Tensor,
        event_mask: Tensor,
        summary: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return `(summaries [B, N, D], zeroed on padding events; token states [B, N, T, D])`."""
        batch, n_events, n_tok, dim = tokens.shape
        evt = _summary(self.evt, summary, "[EVT]").view(1, 1, 1, dim)
        evt = evt.expand(batch, n_events, 1, dim)
        rows = torch.cat([evt, tokens], dim=2).reshape(batch * n_events, n_tok + 1, dim)
        mask = torch.cat([event_mask.unsqueeze(-1), token_mask], dim=-1)
        mask = mask.reshape(batch * n_events, n_tok + 1)
        if self.varlen:
            packed, packed_mask, index = compress_rows(rows, mask)
            self.last_padding_fraction = index.padding_fraction()
            if index.n_kept == 0:
                encoded = torch.zeros_like(rows)
            else:
                encoded = scatter_rows(self._run(packed, packed_mask), index, like=rows)
        else:
            # Empty rows get an all-visible mask so softmax stays defined; their
            # outputs are zeroed below.
            empty = mask.sum(dim=-1, keepdim=True) == 0
            encoded = self._run(rows, mask.masked_fill(empty, 1))
        encoded = encoded.view(batch, n_events, n_tok + 1, dim)
        summaries = encoded[:, :, 0, :] * event_mask.unsqueeze(-1).to(encoded.dtype)
        return summaries, encoded[:, :, 1:, :]


class KVTProfileEncoder(nn.Module):
    """`ProfileEncoder`: bidirectional over profile tokens, `[USR]` slot as the summary.

    `[USR]` is always visible, so an empty profile yields a constant. As with
    `KVTEventEncoder`, it is learned or passed as `summary`.
    """

    def __init__(
        self, spec: KVTEncoderSpec, attention: AttentionSpec, *, learned_summary: bool = True
    ) -> None:
        super().__init__()
        self.blocks = encoder_blocks(
            spec, attention, spec.profile_depth, use_rope=True, causal=False
        )
        self.norm = nn.LayerNorm(spec.hidden_size)
        self.usr = nn.Parameter(torch.randn(spec.hidden_size) * 0.02) if learned_summary else None

    def forward(
        self, tokens: Tensor, mask: Tensor, time_log: Tensor, summary: Tensor | None = None
    ) -> Tensor:
        batch, _n, dim = tokens.shape
        usr = _summary(self.usr, summary, "[USR]").view(1, 1, dim).expand(batch, 1, dim)
        hidden = torch.cat([usr, tokens], dim=1)
        full_mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
        positions = torch.cat([torch.zeros_like(time_log[:, :1]), time_log], dim=1)
        for block in self.blocks:
            hidden = block(hidden, full_mask, rope_positions=positions)
        return self.norm(hidden)[:, 0, :]


__all__ = [
    "CalendarEmbedding",
    "EncoderBlock",
    "KVTEncoderSpec",
    "KVTEventEncoder",
    "KVTProfileEncoder",
    "encoder_blocks",
    "sinusoidal_position",
]
