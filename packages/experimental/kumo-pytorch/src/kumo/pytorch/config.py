# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, explicit configuration objects for the bundled reference models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "pragma_s"
    d_model: int = 192
    d_ffn: int = 768
    attention_heads: int = 3
    profile_encoder_layers: int = 1
    event_encoder_layers: int = 5
    history_encoder_layers: int = 2
    activation: str = "gelu"
    normalization: str = "pre_norm"
    dropout: float = 0.1
    event_encoder_layout: str = "padded"

    def validate(self) -> None:
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if (self.d_model // self.attention_heads) % 2:
            raise ValueError("attention head width must be even for rotary embeddings")
        if self.activation != "gelu" or self.normalization != "pre_norm":
            raise ValueError("only GELU with pre-normalization is supported")
        if self.event_encoder_layout not in {"padded", "packed"}:
            raise ValueError("event_encoder_layout must be 'padded' or 'packed'")


@dataclass(frozen=True)
class MaskingConfig:
    token_probability: float = 0.15
    event_probability: float = 0.10
    semantic_type_probability: float = 0.10
    unknown_probability: float = 0.10
    combine: str = "union"
    ensure_at_least_one: bool = True
    supervise_unknown: bool = False
    fields: tuple[str, ...] = ()
