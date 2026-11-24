# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Future-subgraph aggregate forecasting, with separate detached target tensors."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from tfm.contracts.graph import FutureSubgraphTargets


@dataclass(frozen=True)
class FutureSubgraph:
    """Predict standardized signed-log aggregates in (t,t+H] from a graph at t.

    Required output: `predictions['future_subgraph']` with shape `[B,K]`. Horizon and delay are
    mandatory target metadata; a regression head and GNN receive gradients.
    Membership is sampled at t+H only in label construction. Future rows never
    enter the encoder. Train-only target normalization is fitted upstream.
    Wholly masked batches have differentiable zero loss and no optimizer update.
    """

    name: str = field(default="future_subgraph", init=False)
    support_tier: str = field(default="experimental", init=False)

    def loss(self, prediction: Tensor, targets: FutureSubgraphTargets) -> Tensor:
        targets.assert_validated()
        if prediction.shape != targets.values.shape:
            raise ValueError("FutureSubgraph prediction shape differs from targets")
        # Fixed-shape masking keeps invalid target NaNs out of autograd without
        # data-dependent indexing or scalar reads on the training device.
        target = torch.where(targets.valid, targets.values, 0.0)
        diff = torch.where(targets.valid, prediction - target, 0.0)
        return diff.square().sum() / targets.valid.sum().clamp_min(1)

    def target_level(self) -> str:
        return "sequence"

    def horizon_semantics(self) -> str:
        return "target.horizon_us: terminal event times in (cutoff, cutoff+horizon]"

    def cutoff_semantics(self) -> str:
        return "encoder availability <= cutoff; label membership <= cutoff+horizon"

    def gradient(self) -> str:
        return "regression head and GNN only; no gradients into labels or target transforms"

    def leakage_checks(self) -> str:
        return "separate graph/target batches, complete horizons, availability bounds, train-only transforms"

    def required_outputs(self) -> tuple[str, ...]:
        return ("embeddings", "predictions", "loss", "losses")
