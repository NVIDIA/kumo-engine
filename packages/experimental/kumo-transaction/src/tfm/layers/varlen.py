# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Padding-free row handling for the profile/history stack.

The event encoder runs one attention block per event, so a padded KVT batch is
reshaped to `[B * n_events, n_tokens, d]`. Real histories are heavy-tailed: one
user with 500 events pads every other row in the batch out to 500, and each
padded event row is a full attention call over nothing. The same applies inside
an event, where `n_tokens` is the batch-wide maximum.

This module drops those rows and trims that width before attention, then
scatters the results back into the padded layout. It is arithmetic bookkeeping,
not a kernel: the attention call is unchanged, so results are identical to the
padded path up to floating-point reassociation. `cu_seqlens` is exposed because
that is the layout a true varlen kernel (flash-attn) consumes, and this keeps
the boundary information in one place.

The padded path remains the correctness baseline and is what CPU tests compare
against.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class VarlenIndex:
    """Where compressed rows came from, and how long each one is."""

    rows: Tensor
    """Indices into the flattened `[B * n_events]` row axis. Shape [n_kept]."""
    lengths: Tensor
    """Valid token count per kept row. Shape [n_kept]."""
    n_rows: int
    """Row count of the original padded layout."""
    width: int
    """Token width of the original padded layout."""

    @property
    def n_kept(self) -> int:
        return int(self.rows.numel())

    def cu_seqlens(self) -> Tensor:
        """Exclusive prefix sum of `lengths`, as a varlen kernel expects."""
        zero = torch.zeros(1, dtype=torch.int32, device=self.lengths.device)
        return torch.cat([zero, self.lengths.to(torch.int32).cumsum(0, dtype=torch.int32)])

    def padding_fraction(self) -> float:
        """Share of the original `[n_rows, width]` grid that carried no token."""
        capacity = self.n_rows * self.width
        if capacity == 0:
            return 0.0
        return 1.0 - float(self.lengths.sum()) / capacity


def compress_rows(hidden: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, VarlenIndex]:
    """Drop all-padding rows and trim the token width to the longest kept row.

    `hidden` is `[R, T, D]` and `mask` is `[R, T]`. Returns the compressed
    hidden states, their mask, and the index needed by `scatter_rows`.

    Rows must be left-aligned: a row's valid tokens are its first `mask.sum()`
    positions. The KVT collate guarantees this.
    """
    if hidden.ndim != 3:
        raise ValueError(f"compress_rows expects [R, T, D], got {tuple(hidden.shape)}.")
    lengths = mask.bool().sum(dim=-1)
    rows = lengths.nonzero(as_tuple=False).flatten()
    index = VarlenIndex(
        rows=rows,
        lengths=lengths[rows],
        n_rows=int(hidden.shape[0]),
        width=int(hidden.shape[1]),
    )
    if index.n_kept == 0:
        empty = hidden[:0, :1]
        return empty, mask[:0, :1], index
    width = int(index.lengths.max())
    return hidden[rows, :width], mask[rows, :width], index


def scatter_rows(compressed: Tensor, index: VarlenIndex, *, like: Tensor) -> Tensor:
    """Place compressed rows back into the padded `[R, T, D]` layout of `like`.

    Dropped rows and trimmed positions are zero, which is what the padded path
    produces for them once the caller re-applies its masks.
    """
    out = torch.zeros_like(like)
    if index.n_kept == 0:
        return out
    width = int(compressed.shape[1])
    out[index.rows, :width] = compressed.to(out.dtype)
    return out
