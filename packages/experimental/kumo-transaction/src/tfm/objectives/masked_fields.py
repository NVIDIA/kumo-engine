# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BERT-style masked field modelling over flat field-token rows (flat-encoder guide)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from tfm.contracts.errors import IncompatibleCompositionError


@dataclass(frozen=True)
class MaskedFields:
    """Predict hidden field tokens from every other token in the row.

    Label horizon: none. Targets are the row's own input tokens, so no label,
    outcome, or future event is read; pretraining rows end at or before the
    train cutoff like every other objective's. Each field token (never `<bos>`,
    `<sep>`, `<eos>`, or padding) is chosen with `p_field`, and every field of an
    event with `p_event`, so some rows must reconstruct a whole transaction from
    its neighbours. A chosen token becomes `<mask>` with `p_mask_token`, a random
    value of the same field with `p_random_token`, and is kept otherwise.
    Cross-entropy is taken on chosen positions only, over the target field's ids.
    """

    name: str = "masked_fields"
    p_field: float = 0.15
    p_event: float = 0.10
    p_mask_token: float = 0.8
    p_random_token: float = 0.1
    ignore_index: int = -100
    support_tier: str = "experimental"

    def __post_init__(self) -> None:
        for key in ("p_field", "p_event", "p_mask_token", "p_random_token"):
            value = getattr(self, key)
            if not 0.0 <= value <= 1.0:
                raise IncompatibleCompositionError(f"{key} must be in [0, 1], got {value}.")
        if self.p_field == 0.0 and self.p_event == 0.0:
            raise IncompatibleCompositionError("MaskedFields needs p_field or p_event above 0.")
        if self.p_mask_token + self.p_random_token > 1.0:
            raise IncompatibleCompositionError(
                "p_mask_token + p_random_token must not exceed 1; the rest are kept."
            )

    def required_outputs(self) -> tuple[str, ...]:
        return ("logits",)

    def target_level(self) -> str:
        return "event"


def apply_field_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    field_of_id: Tensor,
    field_ranges: Tensor,
    *,
    sep_id: int,
    mask_id: int,
    spec: MaskedFields,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """`(masked input ids, targets)`; targets are `ignore_index` off the chosen positions.

    `field_of_id[id]` is a token's field index (-1 for special tokens) and
    `field_ranges[f]` is field `f`'s `[start, stop)` id range.
    """
    device = input_ids.device
    field = field_of_id[input_ids]
    candidate = (field >= 0) & attention_mask.bool()
    chosen = candidate & (
        torch.rand(input_ids.shape, generator=generator, device=device) < spec.p_field
    )
    if spec.p_event > 0.0:
        event = torch.cumsum((input_ids == sep_id).long(), dim=1)
        n_events = int(event.max()) + 1
        drawn = torch.rand((input_ids.shape[0], n_events), generator=generator, device=device)
        chosen |= candidate & torch.gather(drawn < spec.p_event, 1, event)
    if not bool(chosen.any()) and bool(candidate.any()):
        # A tiny batch can draw nothing; choose one token so the loss is defined.
        chosen.view(-1)[candidate.view(-1).nonzero()[0]] = True
    targets = torch.where(chosen, input_ids, torch.full_like(input_ids, spec.ignore_index))
    draw = torch.rand(input_ids.shape, generator=generator, device=device)
    to_mask = chosen & (draw < spec.p_mask_token)
    to_random = (
        chosen & (draw >= spec.p_mask_token) & (draw < spec.p_mask_token + spec.p_random_token)
    )
    safe_field = field.clamp(min=0)
    start = field_ranges[safe_field, 0]
    width = field_ranges[safe_field, 1] - start
    offset = torch.rand(input_ids.shape, generator=generator, device=device) * width
    random_ids = start + offset.long().clamp(max=width - 1)
    masked = torch.where(to_mask, torch.full_like(input_ids, mask_id), input_ids)
    masked = torch.where(to_random, random_ids, masked)
    return masked, targets


def masked_fields_loss(
    logits: Tensor,
    targets: Tensor,
    field_of_id: Tensor,
    field_allowed: Tensor,
) -> Tensor:
    """Cross-entropy over each target's own field.

    `logits` and `targets` are the chosen positions only, `[N, V]` and `[N]`;
    `field_allowed[f]` is a `[V]` mask of field `f`'s ids.
    """
    allowed = field_allowed[field_of_id[targets]]
    scores = logits.float().masked_fill(~allowed, float("-inf"))
    return F.cross_entropy(scores, targets)
