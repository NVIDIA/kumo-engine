# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper-based decoder sizes and explicit reference implementation choices."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NuFormerConfig:
    vocab_size: int
    dim: int = 256
    depth: int = 24
    n_heads: int = 16
    ffn_multiplier: int = 4
    dropout: float = 0.0
    bias: bool = True
    layer_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    max_seq_len: int = 2048
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        for name in ("vocab_size", "dim", "depth", "n_heads", "ffn_multiplier", "max_seq_len"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not self.layer_norm_eps > 0:
            raise ValueError("layer_norm_eps must be positive")
        if not isinstance(self.pad_token_id, int) or not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be an integer in the vocabulary")
