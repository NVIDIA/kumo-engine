# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The churn model, its label, and the JSON writer for this example's files."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kumo.pytorch import api


class ChurnModel(nn.Module):
    """Customer features plus a pooled purchase history, into one churn logit."""

    def __init__(
        self,
        output_schema: Mapping[str, Any],
        numbers: tuple[str, ...],
        categories: tuple[str, ...],
    ) -> None:
        super().__init__()
        customer, history = output_schema["inputs"]["customer"], output_schema["inputs"]["history"]
        self.numbers, self.category_names = numbers, categories
        self.age = nn.Embedding(customer["age"]["vocabulary_size"], 4)
        self.categories = nn.ModuleDict(
            {name: nn.Embedding(customer[name]["vocabulary_size"], 4) for name in categories}
        )
        self.channel = nn.Embedding(history["channel"]["vocabulary_size"], 4)
        self.weekday = nn.Embedding(history["weekday"]["vocabulary_size"], 4)
        self.event = nn.Linear(2 + 4 + 4, 16)
        width = 4 + 2 * len(numbers) + 4 * len(categories) + 16
        self.head = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, batch: api.Batch) -> torch.Tensor:
        customer, history = batch.inputs["customer"], batch.inputs["history"]
        numbers = [customer[name]["values"].float().nan_to_num() for name in self.numbers]
        present = [customer[name]["valid"].float() for name in self.numbers]
        events = torch.relu(
            self.event(
                torch.cat(
                    (
                        history["price"]["values"].float().unsqueeze(-1),
                        history["age_days"]["values"].float().unsqueeze(-1),
                        self.channel(history["channel"]["values"]),
                        self.weekday(history["weekday"]["values"]),
                    ),
                    -1,
                )
            )
        )
        valid = history["valid"].unsqueeze(-1).float()
        pooled = (events * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        features = torch.cat(
            (
                self.age(customer["age"]["values"]),
                torch.stack(numbers + present, -1),
                *(self.categories[name](customer[name]["values"]) for name in self.category_names),
                pooled,
            ),
            -1,
        )
        return self.head(features).squeeze(-1)


def churned(batch: api.Batch) -> torch.Tensor:
    """1 where the customer makes no purchase in the label window."""

    return (batch.targets["purchases"]["values"] == 0).float()


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
    """Sorted, indented JSON; datetimes as ISO 8601."""

    def encode(item: object) -> str:
        if isinstance(item, dt.datetime | dt.date):
            return item.isoformat()
        raise TypeError(f"cannot write {type(item).__name__} as JSON")

    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=encode)
    Path(path).write_text(text + "\n")
