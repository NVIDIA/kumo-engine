# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model container for causal token sequences."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class TokenSequenceBatch:
    """A padded causal sequence plus optional profile and pairing information."""

    entity_ids: torch.Tensor
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    token_mask: torch.Tensor
    profile_ids: torch.Tensor
    profile_mask: torch.Tensor
    contrastive_groups: torch.Tensor

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> TokenSequenceBatch:
        return TokenSequenceBatch(
            **{
                field.name: getattr(self, field.name).to(device, non_blocking=non_blocking)
                for field in fields(self)
            }
        )

    @property
    def batch_size(self) -> int:
        return int(self.entity_ids.numel())

    @property
    def token_count(self) -> int:
        return int(self.token_mask.sum().item())
