# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The transaction encoder, its pre-training loss, the fraud head and this example's JSON files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from kumo.pytorch import api

WIDTH = 48
PROFILE_FIELDS = 5
SUMMARY = (PROFILE_FIELDS + 2) * WIDTH


class TransactionEncoder(nn.Module):
    """Averages each transaction's tokens, then a Transformer over profile and transactions.

    Returns a per-user summary and a per-token context in the packed token order.
    """

    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.tokens = nn.Embedding(vocabulary_size, WIDTH)
        self.elapsed = nn.Linear(1, WIDTH)
        layer = nn.TransformerEncoderLayer(
            WIDTH, 4, 2 * WIDTH, 0.1, batch_first=True, norm_first=True
        )
        self.history = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def forward(self, batch: api.Batch) -> tuple[torch.Tensor, torch.Tensor]:
        profile, history = batch.inputs["profile"], batch.inputs["history"]
        sequence = history["sequence"]
        offsets = history["field_offsets"]
        events = offsets.numel() - 1
        event_of = torch.repeat_interleave(torch.arange(events), offsets.diff())
        values = self.tokens(history["value_ids"]) + self.tokens(history["field_ids"])
        values = values * history["valid"].unsqueeze(-1)
        event_vectors = torch.zeros(events, WIDTH).index_add(0, event_of, values)
        event_vectors = event_vectors / offsets.diff().clamp(min=1).unsqueeze(-1)

        age = batch.inputs["history_timing"]["age_days"]["values"].float().nan_to_num()
        slots = event_vectors[sequence["event_indices"].clamp(min=0)]
        slots = slots + self.elapsed(torch.log1p(age.clamp(min=0)).unsqueeze(-1))
        slots = slots * sequence["valid"].unsqueeze(-1)
        people = self.tokens(profile["value_ids"]) + self.tokens(profile["field_ids"])
        size = people.shape[0]
        x = torch.cat((people, slots), 1)
        keep = torch.cat((profile["valid"], sequence["valid"]), 1)
        out = self.history(x, src_key_padding_mask=~keep)
        in_slots = out[:, people.shape[1] :]
        context = torch.zeros(events, WIDTH).index_copy(
            0, sequence["event_indices"][sequence["valid"]], in_slots[sequence["valid"]]
        )
        mean = (out * keep.unsqueeze(-1)).sum(1) / keep.sum(1, keepdim=True).clamp(min=1)
        latest = sequence["valid"].sum(1).sub(1).clamp(min=0)
        last = in_slots[torch.arange(size), latest] * sequence["valid"].any(1, keepdim=True)
        profile_context = out[:, : people.shape[1]].flatten(1)
        return torch.cat((profile_context, mean, last), -1), values + context[event_of]


def masked_value_loss(encoder: TransactionEncoder, batch: api.Batch) -> torch.Tensor:
    """Cross-entropy of each masked token's original, scored with the tied token embedding."""

    _, token_context = encoder(batch)
    target = batch.targets["masked"]
    logits = token_context[target["valid"]] @ encoder.tokens.weight.T / WIDTH**0.5
    return F.cross_entropy(logits, target["values"][target["valid"]])


class FraudModel(nn.Module):
    """A frozen pre-trained encoder and a new head on its user summary."""

    def __init__(self, encoder: TransactionEncoder) -> None:
        super().__init__()
        self.encoder = encoder.requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(SUMMARY, WIDTH), nn.GELU(), nn.Linear(WIDTH, 1))

    def forward(self, batch: api.Batch) -> torch.Tensor:
        summary, _ = self.encoder(batch)
        return self.head(summary).squeeze(-1)


def is_fraud(batch: api.Batch) -> torch.Tensor:
    return (batch.targets["fraud"]["values"] > 0).float()


def rounded(value: Any, digits: int = 6) -> Any:
    """``value`` with every float kept to ``digits`` significant digits."""

    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, Mapping):
        return {str(key): rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(item, digits) for item in value]
    return value


def write_json(path: str, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
