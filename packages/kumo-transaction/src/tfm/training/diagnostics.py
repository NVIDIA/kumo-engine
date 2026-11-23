# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label-alignment diagnostics for causal language models.

A causal model trained with unshifted labels can see the token it is asked to
predict, so it learns to copy its input and its training loss still looks
healthy. That is what happened to the published v1 checkpoint (ADR 0010). The
check that exposes it compares two cross-entropies over the same logits: one
against the next token, which is what the model should predict, and one against
the token at the same position, which a copying model predicts instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class AlignmentLosses:
    next_token: float
    """Mean cross-entropy of position *i*'s logits against token *i + 1*."""
    same_token: float
    """Mean cross-entropy of position *i*'s logits against token *i*."""
    next_token_accuracy: float
    same_token_accuracy: float
    positions: int
    """Scored positions: valid at both *i* and *i + 1*."""

    @property
    def learned_to_copy(self) -> bool:
        return self.same_token <= self.next_token


def label_alignment_losses(
    logits: Tensor, input_ids: Tensor, attention_mask: Tensor
) -> AlignmentLosses:
    """Next-token and same-token losses over the positions both are defined on.

    Both are scored on the same positions, so the comparison cannot be skewed
    by the last valid token, which has no next token.
    """
    if logits.shape[:2] != input_ids.shape or input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)}, input_ids {tuple(input_ids.shape)}, and "
            f"attention_mask {tuple(attention_mask.shape)} must agree on (batch, length)."
        )
    scored = logits[:, :-1, :].float()
    valid = (attention_mask[:, :-1] > 0) & (attention_mask[:, 1:] > 0)
    flat_logits = scored[valid]
    next_targets = input_ids[:, 1:][valid]
    same_targets = input_ids[:, :-1][valid]
    positions = int(valid.sum())
    if positions == 0:
        raise ValueError("No position has a valid next token to score.")
    with torch.no_grad():
        predicted = flat_logits.argmax(dim=-1)
        return AlignmentLosses(
            next_token=float(F.cross_entropy(flat_logits, next_targets)),
            same_token=float(F.cross_entropy(flat_logits, same_targets)),
            next_token_accuracy=float((predicted == next_targets).float().mean()),
            same_token_accuracy=float((predicted == same_targets).float().mean()),
            positions=positions,
        )
