# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mean over the last event's field tokens (experimental)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LastEventMeanPool:
    """Average the states of the row's last event. Not a core export.

    The bidirectional analogue of `LastTokenPool`: a probe row ends at the
    transaction it scores, so the last event's tokens are that transaction read
    in the context of every earlier one. Only `FlatEncoder` accepts it, because
    it needs field-token positions, which the model supplies.
    """

    name: str = "last_event_mean"
    support_tier: str = "experimental"


def last_event_mask(
    input_ids: Tensor, attention_mask: Tensor, field_of_id: Tensor, sep_id: int
) -> Tensor:
    """`[B, S]` true at the field tokens of each row's last event.

    `field_of_id[id]` is the field index of a token id, or -1 for special tokens.
    A row with no field token selects nothing.
    """
    is_field = (field_of_id[input_ids] >= 0) & attention_mask.bool()
    event = torch.cumsum((input_ids == sep_id).long(), dim=1)
    last = torch.where(is_field, event, torch.full_like(event, -1)).amax(dim=1, keepdim=True)
    return is_field & (event == last)
