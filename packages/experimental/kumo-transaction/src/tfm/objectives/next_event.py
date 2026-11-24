# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal next-event (next-token) objective."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from tfm.contracts.batch import TransactionBatch
from tfm.contracts.outputs import ModelOutput


@dataclass(frozen=True)
class NextEvent:
    name: str = "causal_next_event"
    ignore_index: int = -100

    def required_outputs(self) -> tuple[str, ...]:
        return ("logits",)

    def target_level(self) -> str:
        return "event"


class NextEventLoss(nn.Module):
    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.crit = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, output: ModelOutput, batch: TransactionBatch) -> Tensor:
        if output.logits is None:
            raise ValueError("NextEvent requires logits on ModelOutput.")
        logits = output.logits[:, :-1, :].contiguous()
        if batch.labels is not None:
            targets = batch.labels[:, 1:].contiguous()
        else:
            targets = batch.input_ids[:, 1:].contiguous()
            pad = batch.attention_mask[:, 1:] == 0
            targets = targets.clone()
            targets[pad] = self.ignore_index
        return self.crit(logits.view(-1, logits.size(-1)), targets.view(-1))
