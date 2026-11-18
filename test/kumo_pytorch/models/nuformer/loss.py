# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Next-token loss for complete, unshifted user sequences."""

import torch
from torch.nn import functional as F

from .model import _check_tokens_and_mask


def next_token_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    ignore_index: int = -100,
    validate_inputs: bool = True,
) -> torch.Tensor:
    """Mean CE over adjacent valid tokens, shifting targets exactly once here.

    logits[:, t] predicts input_ids[:, t+1]. Optional labels have the SAME
    positions as input_ids and can mark excluded targets with ignore_index.
    Never pass targets that have already been shifted by a data loader.
    Histories with fewer than two valid tokens contribute no targets. If no
    row contributes, the result is a graph-connected zero with zero gradients.
    An explicit mask is required so loss and model use the same padding policy.
    """
    _check_tokens_and_mask(input_ids, attention_mask, validate_inputs=validate_inputs)
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape or logits.shape[-1] < 2:
        raise ValueError("logits must have shape [batch, length, vocabulary] matching input_ids")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if logits.device != input_ids.device:
        raise ValueError("logits and input tensors must be on the same device")
    if labels is None:
        labels = input_ids
    if labels.shape != input_ids.shape or labels.dtype != torch.int64:
        raise ValueError("labels must be unshifted int64 [batch, length] matching input_ids")
    if labels.device != input_ids.device:
        raise ValueError("labels and input_ids must be on the same device")

    targets = labels[:, 1:]
    valid = attention_mask[:, :-1] & attention_mask[:, 1:] & (targets != ignore_index)
    targets = targets.masked_fill(~valid, ignore_index)
    losses = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    return losses.sum() / valid.sum().clamp_min(1)
