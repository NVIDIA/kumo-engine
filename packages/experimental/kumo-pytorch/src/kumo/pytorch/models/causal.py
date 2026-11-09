# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable causal token-sequence encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from kumo.pytorch.models.layers import TransformerStack
from kumo.pytorch.models.protocols import ModelState, model_inputs


@dataclass(frozen=True)
class CausalEncoderConfig:
    d_model: int = 256
    d_ffn: int = 1_024
    attention_heads: int = 4
    layers: int = 4
    dropout: float = 0.1

    def validate(self) -> None:
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")


class CausalTokenEncoder(nn.Module):
    """NoPE Transformer encoder for padded causal token sequences."""

    def __init__(
        self,
        config: CausalEncoderConfig,
        *,
        vocabulary_size: int,
        pad_id: int,
        profile_vocabulary_size: int = 0,
        profile_mix: float = 0.1,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(vocabulary_size, config.d_model, padding_idx=pad_id)
        self.profile_embedding = (
            nn.Embedding(profile_vocabulary_size, config.d_model, padding_idx=0)
            if profile_vocabulary_size
            else None
        )
        self.profile_mix = profile_mix
        self.output_width = config.d_model
        self.transformer = TransformerStack(
            config.layers,
            config.d_model,
            config.d_ffn,
            config.attention_heads,
            config.dropout,
        )

    @property
    def output_embedding(self) -> nn.Embedding:
        """Embedding table reused by tied-token prediction objectives."""

        return self.embedding

    @staticmethod
    def last_token(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = mask.sum(dim=1).to(torch.long) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]

    def embed(self, batch: object) -> torch.Tensor:
        """Build token inputs, including optional profile conditioning."""

        inputs = self.embedding(batch.input_ids)
        if self.profile_embedding is None:
            return inputs
        if not bool(batch.profile_mask.any()):
            raise ValueError("profile-conditioned encoder requires profile IDs")
        profile_tokens = self.profile_embedding(batch.profile_ids)
        count = batch.profile_mask.sum(dim=1, keepdim=True).clamp_min(1)
        profile = (profile_tokens * batch.profile_mask.unsqueeze(-1)).sum(dim=1) / count
        inputs = (1.0 - self.profile_mix) * inputs + self.profile_mix * profile[:, None]
        inputs[:, 0] = profile
        return inputs

    def forward(self, wrapped_batch: object) -> ModelState:
        batch = model_inputs(wrapped_batch)
        hidden = self.transformer(
            self.embed(batch),
            batch.token_mask,
            causal=True,
        )
        return ModelState(
            contextual=hidden,
            mask=batch.token_mask,
            representations={"entity": self.last_token(hidden, batch.token_mask)},
            auxiliary={},
        )
