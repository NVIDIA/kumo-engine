# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary classification head (linear logit)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class Binary:
    name: str = "binary"
    target_level: str = "sequence"


class BinaryHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.proj(embeddings).squeeze(-1)
