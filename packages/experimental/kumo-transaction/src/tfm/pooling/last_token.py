# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Last valid (non-pad) token pooling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LastTokenPool:
    name: str = "last_token"


class LastTokenPoolModule(nn.Module):
    def forward(self, states: Tensor, attention_mask: Tensor) -> Tensor:
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch = torch.arange(states.size(0), device=states.device)
        return states[batch, lengths]
