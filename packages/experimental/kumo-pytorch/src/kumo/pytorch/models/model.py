# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One model wrapper used by every architecture specification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from kumo.pytorch.models.protocols import ModelOutput, ModelState, ObjectiveOutput


class MultiObjectiveModel(nn.Module):
    """Combine one encoder with any number of weighted training objectives."""

    def __init__(
        self,
        encoder: nn.Module,
        objectives: Mapping[str, nn.Module],
        objective_weights: Mapping[str, float] | None = None,
    ):
        super().__init__()
        if not objectives:
            raise ValueError("at least one objective is required")
        self.encoder = encoder
        self.objectives = nn.ModuleDict(objectives)
        self.objective_weights = {
            name: float((objective_weights or {}).get(name, 1.0)) for name in objectives
        }

    def encode(self, batch: Any) -> ModelState:
        return self.encoder(batch)

    def forward(self, batch: Any) -> ModelOutput:
        state = self.encode(batch)
        losses: dict[str, torch.Tensor] = {}
        predictions: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name, objective in self.objectives.items():
            result: ObjectiveOutput = objective(state, batch)
            losses[name] = result.loss
            losses.update(result.components)
            predictions.update(result.predictions)
            weighted = self.objective_weights[name] * result.loss
            total = weighted if total is None else total + weighted
        if total is None:  # pragma: no cover - guarded by __init__
            raise RuntimeError("model has no objectives")
        return ModelOutput(total, losses, predictions, state.representations)
