# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal encoding of fixed-schema numerical and categorical vectors."""

from __future__ import annotations

import torch
from torch import nn

from kumo.pytorch.data.typed_vectors import AttributeSchema, TypedVectorBatch
from kumo.pytorch.models.causal import CausalEncoderConfig
from kumo.pytorch.models.layers import TransformerStack
from kumo.pytorch.models.protocols import ModelState, model_inputs


def log_scale(values: torch.Tensor) -> torch.Tensor:
    """Compress long-tailed values while preserving the sign if one is present."""

    return values.sign() * torch.log1p(values.abs())


class TypedAttributeEncoder(nn.Module):
    """Project one fixed numerical/categorical vector to a shared width."""

    def __init__(self, schema: AttributeSchema, width: int, categorical_width: int):
        super().__init__()
        if categorical_width <= 0:
            raise ValueError("categorical embedding width must be positive")
        self.schema = schema
        self.width = width
        self.numerical = nn.Linear(2 * len(schema.numerical), width) if schema.numerical else None
        self.categorical = nn.ModuleList(
            nn.Embedding(schema.cardinality(name), categorical_width) for name in schema.categorical
        )
        combined_width = (width if self.numerical is not None else 0) + (
            categorical_width * len(self.categorical)
        )
        if combined_width == 0:
            raise ValueError("typed attribute encoder requires at least one field")
        self.projection = nn.Sequential(
            nn.Linear(combined_width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(
        self,
        numerical: torch.Tensor,
        numerical_present: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        pieces = []
        if self.numerical is not None:
            numerical_input = torch.cat(
                (log_scale(numerical), numerical_present.to(numerical.dtype)),
                dim=-1,
            )
            pieces.append(self.numerical(numerical_input))
        pieces.extend(
            embedding(categorical[..., index]) for index, embedding in enumerate(self.categorical)
        )
        return self.projection(torch.cat(pieces, dim=-1))


class TypedVectorEncoder(nn.Module):
    """Place a static vector before dynamic vectors in a causal Transformer."""

    def __init__(
        self,
        config: CausalEncoderConfig,
        static_schema: AttributeSchema,
        dynamic_schema: AttributeSchema,
        *,
        categorical_embedding_width: int = 32,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.static = TypedAttributeEncoder(
            static_schema,
            config.d_model,
            categorical_embedding_width,
        )
        self.dynamic = TypedAttributeEncoder(
            dynamic_schema,
            config.d_model,
            categorical_embedding_width,
        )
        self.transformer = TransformerStack(
            config.layers,
            config.d_model,
            config.d_ffn,
            config.attention_heads,
            config.dropout,
        )
        self.output_width = config.d_model

    def forward(self, wrapped_batch: object) -> ModelState:
        batch = model_inputs(wrapped_batch)
        if not isinstance(batch, TypedVectorBatch):
            raise TypeError("typed vector encoder requires TypedVectorBatch")
        static = self.static(
            batch.static_numerical,
            batch.static_numerical_present,
            batch.static_categorical,
        ).unsqueeze(1)
        dynamic = self.dynamic(
            batch.dynamic_numerical,
            batch.dynamic_numerical_present,
            batch.dynamic_categorical,
        )
        inputs = torch.cat((static, dynamic), dim=1)
        mask = torch.cat(
            (
                torch.ones(
                    (batch.batch_size, 1),
                    dtype=torch.bool,
                    device=batch.event_mask.device,
                ),
                batch.event_mask,
            ),
            dim=1,
        )
        hidden = self.transformer(inputs, mask, causal=True)
        dynamic_hidden = hidden[:, 1:]
        positions = batch.event_mask.sum(dim=1).to(torch.long) - 1
        entity = dynamic_hidden[
            torch.arange(batch.batch_size, device=dynamic_hidden.device), positions
        ]
        return ModelState(
            contextual=dynamic_hidden,
            mask=batch.event_mask,
            representations={"entity": entity, "static": hidden[:, 0]},
            auxiliary={},
        )
