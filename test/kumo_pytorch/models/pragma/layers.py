# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain PyTorch transformer blocks for packed PRAGMA inputs.

Each event, profile, or history is an independent, bidirectional attention
segment. We slice those segments, call PyTorch SDPA, and concatenate the results.
There are no custom kernels, padding buckets, or device-specific execution paths.
Projection names and shapes match the original model's checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .embeddings import TimeRoPE


def _segment_boundaries(
    cu_seqlens: torch.Tensor, n_tokens: int, max_seqlen: int | None,
) -> list[int]:
    """Validate packed offsets, including optional legacy length metadata.

    Reading offsets on the host is deliberate: this reference implementation
    favors clear validation and slicing over avoiding accelerator synchronization.
    """
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must be a one-dimensional integer tensor")
    offsets = cu_seqlens.tolist()
    if not offsets or offsets[0] != 0 or offsets[-1] != n_tokens:
        raise ValueError("cu_seqlens must start at zero and end at the token count")
    lengths = [end - start for start, end in zip(offsets, offsets[1:])]
    if any(length < 0 for length in lengths):
        raise ValueError("cu_seqlens must be nondecreasing")
    if max_seqlen is not None:
        if type(max_seqlen) is not int or max_seqlen < max(lengths, default=0):
            raise ValueError("max_seqlen must be an integer at least as large as the longest segment")
    return offsets


def varlen_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int | None = None,
    rope: TimeRoPE | None = None,
    rope_pos: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Apply bidirectional attention independently to each packed segment.

    q, k, and v have shape [tokens, heads, head_dim]. Offsets partition their
    first dimension; repeated offsets represent empty segments. Optional
    TimeRoPE rotates queries and keys using the supplied time positions.
    The result has the same shape and token order as the inputs.
    """
    if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have matching [tokens, heads, head_dim] shapes")
    if q.shape[1] < 1 or q.shape[2] < 1 or not q.is_floating_point():
        raise ValueError("attention requires floating-point inputs with positive head dimensions")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if k.device != q.device or v.device != q.device or cu_seqlens.device != q.device:
        raise ValueError("attention inputs and offsets must be on the same device")
    offsets = _segment_boundaries(cu_seqlens, q.shape[0], max_seqlen)
    if not 0.0 <= dropout_p <= 1.0:
        raise ValueError("dropout_p must be between zero and one")

    if rope is not None:
        if rope_pos is None or rope_pos.shape != (q.shape[0],):
            raise ValueError("TimeRoPE requires one time position per token")
        if rope_pos.device != q.device or rope.inv_freq.device != q.device:
            raise ValueError("TimeRoPE and positions must be on the attention device")
        cos, sin = rope.angles(rope_pos)
        q = rope.rotate(q.transpose(0, 1), cos, sin).transpose(0, 1)
        k = rope.rotate(k.transpose(0, 1), cos, sin).transpose(0, 1)

    if q.shape[0] == 0:
        # Keep all three inputs connected to autograd even for an empty batch.
        return q + k + v

    outputs = []
    for start, end in zip(offsets, offsets[1:]):
        if start == end:
            continue
        # SDPA expects [batch, heads, length, head_dim]. A segment is one batch
        # row, so attention cannot cross event or user boundaries.
        qs, ks, vs = (t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        attended = F.scaled_dot_product_attention(
            qs, ks, vs, dropout_p=dropout_p, is_causal=False,
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


class VarlenAttention(nn.Module):
    """Multi-head bidirectional attention over packed, independent segments."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1, rope_base: float = 10_000.0,
                 use_rope: bool = False) -> None:
        super().__init__()
        if n_heads < 1 or dim < 1 or dim % n_heads != 0:
            raise ValueError(f"positive dim {dim} must be divisible by positive n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.rope = TimeRoPE(self.head_dim, base=rope_base) if use_rope else None

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int | None = None,
                rope_pos: torch.Tensor | None = None) -> torch.Tensor:
        """Map flat [tokens, dim] representations to the same shape."""
        qkv = self.qkv(x).view(x.shape[0], 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)
        attended = varlen_self_attention(
            q, k, v, cu_seqlens, max_seqlen, self.rope, rope_pos,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(attended.reshape(x.shape[0], self.dim))


class FeedForward(nn.Module):
    """GELU feedforward network with hidden width mult * dim."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mult * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mult * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm attention and feedforward sublayers with residual connections."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 4,
                 use_rope: bool = False, rope_base: float = 10_000.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VarlenAttention(dim, n_heads, dropout, rope_base, use_rope)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int | None = None,
                rope_pos: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), cu_seqlens, max_seqlen, rope_pos))
        return x + self.drop(self.ffn(self.norm2(x)))


class Encoder(nn.Module):
    """A stack of transformer blocks followed by final LayerNorm."""

    def __init__(self, dim: int, depth: int, n_heads: int, dropout: float = 0.1,
                 use_rope: bool = False, rope_base: float = 10_000.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, dropout, 4, use_rope, rope_base) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int | None = None,
                rope_pos: torch.Tensor | None = None) -> torch.Tensor:
        # Validate even a zero-depth encoder, which only applies normalization.
        _segment_boundaries(cu_seqlens, x.shape[0], max_seqlen)
        for block in self.blocks:
            x = block(x, cu_seqlens, max_seqlen, rope_pos)
        return self.norm(x)
