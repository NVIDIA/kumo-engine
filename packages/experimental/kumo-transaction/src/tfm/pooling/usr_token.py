# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""History-encoder [USR] slot pooling (experimental)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class UsrTokenPool:
    """Readout token for profile/event/history stacks. Not a core export.

    Valid only under a bidirectional history model: under a causal mask, slot 0
    has seen only the profile.
    """

    name: str = "usr_token"
    support_tier: str = "experimental"


class UsrTokenPoolModule(nn.Module):
    def forward(self, states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        del attention_mask
        return states[:, 0, :]
