# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Padding and causal mask construction."""

from __future__ import annotations

import torch
from torch import Tensor


def padding_mask(attention_mask: Tensor) -> Tensor:
    """Bool mask of valid (non-pad) positions. Shape [B, T]."""
    return attention_mask.bool()


def causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Lower-triangular causal visibility. Shape [T, T]."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def segment_mask(segment_ids: Tensor) -> Tensor:
    """Block-diagonal visibility for packed sequences. Shape [B, 1, T, T].

    A query may only attend to keys from its own packed segment, so packing
    several entity histories into one row never lets one history see another.
    """
    same = segment_ids[:, :, None] == segment_ids[:, None, :]
    return same[:, None, :, :]


def attention_bias(
    attention_mask: Tensor, *, causal: bool, segment_ids: Tensor | None = None
) -> Tensor:
    """Combined key visibility for eager attention. Shape [B, 1, T, T]."""
    batch, seq_len = attention_mask.shape
    key_valid = attention_mask.bool()[:, None, None, :]
    visible = key_valid if not causal else key_valid & causal_mask(seq_len, attention_mask.device)
    visible = visible.expand(batch, 1, seq_len, seq_len)
    if segment_ids is not None:
        visible = visible & segment_mask(segment_ids)
    return visible
