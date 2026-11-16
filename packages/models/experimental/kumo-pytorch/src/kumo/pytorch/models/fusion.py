# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable parameter-efficient tuning and differentiable tabular fusion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from kumo.pytorch.models.common import LowRankLinear
from kumo.pytorch.models.layers import TransformerStack
from kumo.pytorch.models.protocols import LabeledBatch, ModelState


class TabularFeatureEncoder(nn.Module):
    """Encode numerical and categorical columns without erasing their types.

    Numerical columns use learned periodic features after fixed normalization.
    Each categorical column has its own embedding table. The resulting pieces
    form the input to the shared deep-cross network.
    """

    def __init__(
        self,
        numerical_means: tuple[float, ...],
        numerical_scales: tuple[float, ...],
        periodic_frequencies: int,
        categorical_cardinalities: tuple[int, ...],
        categorical_embedding_width: int,
    ):
        super().__init__()
        if len(numerical_means) != len(numerical_scales):
            raise ValueError("each numerical mean requires one scale")
        if any(scale <= 0 for scale in numerical_scales):
            raise ValueError("numerical scales must be positive")
        if numerical_means and periodic_frequencies <= 0:
            raise ValueError("periodic frequencies must be positive")
        if categorical_cardinalities and categorical_embedding_width <= 0:
            raise ValueError("categorical embedding width must be positive")
        if any(cardinality <= 1 for cardinality in categorical_cardinalities):
            raise ValueError("categorical cardinalities must be at least two")
        if not numerical_means and not categorical_cardinalities:
            raise ValueError("tabular fusion requires at least one feature")

        self.register_buffer("numerical_means", torch.tensor(numerical_means))
        self.register_buffer("numerical_scales", torch.tensor(numerical_scales))
        self.frequencies = nn.Parameter(torch.empty(len(numerical_means), periodic_frequencies))
        self.phases = nn.Parameter(torch.zeros(len(numerical_means), periodic_frequencies))
        if self.frequencies.numel():
            nn.init.normal_(self.frequencies)
        self.categorical = nn.ModuleList(
            nn.Embedding(cardinality, categorical_embedding_width)
            for cardinality in categorical_cardinalities
        )
        self.output_width = (
            2 * len(numerical_means) * periodic_frequencies
            + len(categorical_cardinalities) * categorical_embedding_width
        )
        self.output_norm = nn.LayerNorm(self.output_width)

    def forward(
        self,
        numerical: torch.Tensor | None,
        categorical: torch.Tensor | None,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        batch_size: int | None = None
        if self.numerical_means.numel():
            if numerical is None or numerical.ndim != 2:
                raise ValueError("tabular_numerical must be a two-dimensional tensor")
            if numerical.shape[1] != self.numerical_means.numel():
                raise ValueError("tabular_numerical width does not match its configuration")
            batch_size = numerical.shape[0]
            normalized = (
                numerical.to(self.numerical_means.dtype) - self.numerical_means
            ) / self.numerical_scales
            angles = normalized.unsqueeze(-1) * self.frequencies + self.phases
            pieces.append(torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(1))
        elif numerical is not None:
            raise ValueError("tabular_numerical was supplied but no numerical fields are declared")

        if self.categorical:
            if categorical is None or categorical.ndim != 2:
                raise ValueError("tabular_categorical must be a two-dimensional tensor")
            if categorical.shape[1] != len(self.categorical):
                raise ValueError("tabular_categorical width does not match its configuration")
            if batch_size is not None and categorical.shape[0] != batch_size:
                raise ValueError("tabular numerical and categorical batch sizes differ")
            pieces.extend(
                embedding(categorical[:, index].to(torch.long))
                for index, embedding in enumerate(self.categorical)
            )
        elif categorical is not None:
            raise ValueError(
                "tabular_categorical was supplied but no categorical fields are declared"
            )
        return self.output_norm(torch.cat(pieces, dim=-1))


class CrossLayer(nn.Module):
    """Low-rank DCNv2 cross layer."""

    def __init__(self, width: int, rank: int):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width)

    def forward(self, original: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return original * self.up(F.gelu(self.down(current))) + current


class DeepCrossNetwork(nn.Module):
    """Differentiable tabular branch suitable for joint fusion."""

    def __init__(self, input_width: int, output_width: int, layers: int = 2, rank: int = 32):
        super().__init__()
        if input_width <= 0:
            raise ValueError("tabular input width must be positive")
        rank = min(rank, input_width)
        self.cross_layers = nn.ModuleList(CrossLayer(input_width, rank) for _ in range(layers))
        self.deep = nn.Sequential(
            nn.Linear(input_width, output_width),
            nn.GELU(),
            nn.Linear(output_width, output_width),
        )
        self.output = nn.Linear(input_width + output_width, output_width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        crossed = inputs
        for layer in self.cross_layers:
            crossed = layer(inputs, crossed)
        return self.output(torch.cat((crossed, self.deep(inputs)), dim=-1))


def add_transformer_lora(
    encoder: nn.Module,
    *,
    rank: int,
    dropout: float = 0.1,
) -> None:
    """Freeze an encoder and add LoRA updates to every Transformer stack it owns."""

    stacks = [module for module in encoder.modules() if isinstance(module, TransformerStack)]
    if not stacks:
        raise TypeError("LoRA requires an encoder containing a Transformer stack")
    if any(
        isinstance(block.attention.qkv, LowRankLinear) for stack in stacks for block in stack.blocks
    ):
        raise ValueError("the encoder already contains LoRA adapters")
    encoder.requires_grad_(False)
    for stack in stacks:
        for block in stack.blocks:
            block.attention.qkv = LowRankLinear(block.attention.qkv, rank, dropout)
            block.attention.output = LowRankLinear(block.attention.output, rank, dropout)
            block.ffn[0] = LowRankLinear(block.ffn[0], rank, dropout)
            block.ffn[3] = LowRankLinear(block.ffn[3], rank, dropout)


class JointFusionEncoder(nn.Module):
    """Concatenate a sequence representation with a DCNv2 tabular representation."""

    def __init__(
        self,
        sequence: nn.Module,
        numerical_means: tuple[float, ...],
        numerical_scales: tuple[float, ...],
        periodic_frequencies: int,
        categorical_cardinalities: tuple[int, ...],
        categorical_embedding_width: int,
        *,
        layers: int = 2,
        rank: int = 32,
    ):
        super().__init__()
        if not hasattr(sequence, "output_width"):
            raise TypeError("the sequence encoder must expose output_width")
        self.sequence = sequence
        sequence_width = int(sequence.output_width)
        self.tabular_features = TabularFeatureEncoder(
            numerical_means,
            numerical_scales,
            periodic_frequencies,
            categorical_cardinalities,
            categorical_embedding_width,
        )
        self.tabular = DeepCrossNetwork(
            self.tabular_features.output_width,
            sequence_width,
            layers=layers,
            rank=rank,
        )
        self.output_width = 2 * sequence_width

    def forward(self, batch: object) -> ModelState:
        if not isinstance(batch, LabeledBatch):
            raise TypeError("joint fusion requires LabeledBatch")
        state = self.sequence(batch.inputs)
        features = self.tabular_features(
            batch.features.get("tabular_numerical"),
            batch.features.get("tabular_categorical"),
        ).to(state.representations["entity"].dtype)
        tabular = self.tabular(features)
        entity = torch.cat((state.representations["entity"], tabular), dim=-1)
        return ModelState(
            contextual=state.contextual,
            mask=state.mask,
            representations={**state.representations, "entity": entity, "tabular": tabular},
            auxiliary=state.auxiliary,
        )
