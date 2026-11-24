# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model output contract."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class ModelOutput:
    """Distinguishes token states, pooled embeddings, and task outputs."""

    hidden_states: Tensor
    embeddings: Tensor | None = None
    logits: Tensor | None = None
    loss: Tensor | None = None
    predictions: dict[str, Tensor] | None = None
    losses: dict[str, Tensor] | None = None
    """Per-objective components of `loss` when more than one objective is composed."""
