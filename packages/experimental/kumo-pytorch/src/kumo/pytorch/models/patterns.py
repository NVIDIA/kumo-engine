# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal local-pattern extraction with a learned prototype memory."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from kumo.pytorch.models.causal import CausalTokenEncoder
from kumo.pytorch.models.protocols import ModelState, model_inputs


class PatternMemoryEncoder(nn.Module):
    """Enrich a causal token encoder with local convolutions and prototypes.

    Left padding makes every convolution causal: an output can use the current
    behavior and earlier behaviors, never a future one. Each local pattern then
    attends over a small learned prototype table alongside the Transformer.
    """

    def __init__(
        self,
        sequence: CausalTokenEncoder,
        *,
        prototypes: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4),
        convolution_kernel_size: int = 3,
        initial_pattern_mix: float = 0.1,
    ):
        super().__init__()
        if prototypes <= 0 or not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError("prototypes and convolution dilations must be positive")
        if convolution_kernel_size <= 0:
            raise ValueError("convolution kernel size must be positive")
        self.sequence = sequence
        width = sequence.config.d_model
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                width,
                width,
                convolution_kernel_size,
                dilation=dilation,
                groups=width,
            )
            for dilation in dilations
        )
        self.dilations = dilations
        self.convolution_kernel_size = convolution_kernel_size
        self.pattern_projection = nn.Linear(width * len(dilations), width)
        self.prototypes = nn.Parameter(torch.empty(prototypes, width))
        self.prototype_attention = nn.MultiheadAttention(
            width,
            sequence.config.attention_heads,
            dropout=sequence.config.dropout,
            batch_first=True,
        )
        self.pattern_norm = nn.LayerNorm(width)
        self.pattern_mix = nn.Parameter(torch.tensor(initial_pattern_mix))
        nn.init.normal_(self.prototypes, std=width**-0.5)
        self.output_width = width

    @property
    def output_embedding(self) -> nn.Embedding:
        """Embedding table reused by next-behavior prediction."""

        return self.sequence.embedding

    def forward(self, wrapped_batch: object) -> ModelState:
        batch = model_inputs(wrapped_batch)
        inputs = self.sequence.embed(batch)
        channels = inputs.transpose(1, 2)
        patterns = []
        for dilation, convolution in zip(self.dilations, self.convolutions, strict=True):
            padded = F.pad(
                channels,
                (dilation * (self.convolution_kernel_size - 1), 0),
            )
            patterns.append(convolution(padded).transpose(1, 2))
        local = self.pattern_projection(F.gelu(torch.cat(patterns, dim=-1)))
        memory = self.pattern_norm(self.prototypes)[None].expand(inputs.shape[0], -1, -1)
        recalled, prototype_weights = self.prototype_attention(
            local,
            memory,
            memory,
            need_weights=True,
        )
        transformer_hidden = self.sequence.transformer(inputs, batch.token_mask, causal=True)
        hidden = transformer_hidden + self.pattern_mix.tanh() * self.pattern_norm(recalled)
        hidden = hidden * batch.token_mask.unsqueeze(-1)
        entity = self.sequence.last_token(hidden, batch.token_mask)
        valid_weights = prototype_weights * batch.token_mask.unsqueeze(-1)
        prototype_usage = valid_weights.sum(dim=(0, 1)) / batch.token_mask.sum().clamp_min(1)
        return ModelState(
            contextual=hidden,
            mask=batch.token_mask,
            representations={"entity": entity},
            auxiliary={"prototype_usage": prototype_usage},
        )
