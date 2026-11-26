# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-only controls over the same source-native history deliveries."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from tfm.contracts.batch import TransactionBatch
from tfm.contracts.kvt import KeyValueTimeBatch


@torch.inference_mode(False)
@torch.no_grad()
def last_event_control(batch: Any) -> Any:
    """Keep the scored event; retain full-history raw features on the control arm."""
    if isinstance(batch, TransactionBatch):
        tokens = torch.zeros_like(batch.input_ids)
        mask = torch.zeros_like(batch.attention_mask)
        for i, (row, valid) in enumerate(zip(batch.input_ids, batch.attention_mask, strict=True)):
            row = row[valid.bool()]
            separators = (row == 3).nonzero().flatten()
            event_tokens = row[int(separators[-1]) + 1 :] if len(separators) else row[1:]
            result = torch.cat((row.new_tensor([1]), event_tokens))
            tokens[i, : len(result)] = result
            mask[i, : len(result)] = 1
        return replace(
            batch,
            input_ids=tokens,
            attention_mask=mask,
            labels=tokens.masked_fill(~mask.bool(), -100),
        )
    if not isinstance(batch, KeyValueTimeBatch):
        raise TypeError("Native single-event control requires flat or KVT history tensors")
    valid = batch.event_mask.any(1).unsqueeze(1)
    last = (batch.event_mask.sum(1) - 1).clamp_min(0)

    def event(value: torch.Tensor) -> torch.Tensor:
        return value[torch.arange(batch.batch_size), last].unsqueeze(1).clone()

    return replace(
        batch,
        key_ids=event(batch.key_ids),
        value_ids=event(batch.value_ids),
        positions=event(batch.positions),
        token_mask=event(batch.token_mask),
        event_mask=valid,
        time_log=torch.zeros_like(event(batch.time_log)),
        elapsed_log=torch.zeros_like(event(batch.elapsed_log))
        if batch.elapsed_log is not None
        else None,
        events={name: event(value) for name, value in batch.events.items()},
    )
