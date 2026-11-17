# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The recommendation model, its metric, and the JSON writer for this example's files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import scatter

from kumo.pytorch import api

WIDTH = 32
EDGE_TYPES = (
    ("customers", "customer_transactions", "transactions"),
    ("transactions", "rev_customer_transactions", "customers"),
    ("articles", "article_transactions", "transactions"),
    ("transactions", "rev_article_transactions", "articles"),
)


class Recommender(nn.Module):
    """Two rounds of message passing to a customer vector, scored against each candidate."""

    def __init__(self, output_schema: Mapping[str, Any], article_fields: tuple[str, ...]) -> None:
        super().__init__()
        candidates = output_schema["inputs"]["candidates"]
        customer = output_schema["inputs"]["customer"]
        self.article_fields = article_fields
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(candidates[name]["vocabulary_size"], WIDTH)
                for name in article_fields
            }
        )
        self.club = nn.Embedding(customer["club"]["vocabulary_size"], WIDTH)
        self.customer_in = nn.Linear(1, WIDTH)
        self.purchase_in = nn.Linear(1, WIDTH)
        self.layers = nn.ModuleList(
            HeteroConv({edge: SAGEConv(WIDTH, WIDTH) for edge in EDGE_TYPES}) for _ in range(2)
        )
        self.popularity = nn.Linear(1, 1)

    def articles(self, fields: Mapping[str, Mapping[str, torch.Tensor]]) -> torch.Tensor:
        return sum(self.embeddings[name](fields[name]["values"]) for name in self.article_fields)

    def forward(self, batch: api.Batch) -> torch.Tensor:
        customer, graph, candidates = (
            batch.inputs["customer"],
            batch.inputs["graph"],
            batch.inputs["candidates"],
        )
        nodes = graph["nodes"]
        me = self.customer_in(customer["age"]["values"].float().nan_to_num().unsqueeze(-1))
        me = me + self.club(customer["club"]["values"])
        x = {
            "customers": me[nodes["customers"]["batch"]],
            "transactions": self.purchase_in(
                nodes["transactions"]["features"]["price"]["values"].float().unsqueeze(-1)
            ),
            "articles": self.articles(nodes["articles"]["features"]),
        }
        edges: dict[tuple[str, str, str], torch.Tensor] = {}
        for relation in ("customer_transactions", "article_transactions"):
            index = graph["edges"][relation]["index"]
            source, _, target = next(e for e in EDGE_TYPES if e[1] == relation)
            edges[(source, relation, target)] = index
            edges[(target, f"rev_{relation}", source)] = index.flip(0)
        for layer in self.layers:
            x = {**x, **{name: torch.relu(value) for name, value in layer(x, edges).items()}}
        customers = scatter(
            x["customers"],
            nodes["customers"]["batch"],
            dim=0,
            dim_size=batch.batch_size,
            reduce="mean",
        )
        items = self.articles(candidates)
        popularity = self.popularity(
            candidates["sold_30d"]["values"].float().nan_to_num().unsqueeze(-1)
        )
        return (items * customers.unsqueeze(1)).sum(-1) + popularity.squeeze(-1)


def recall_at(
    k: int, scores: torch.Tensor, bought: torch.Tensor, valid: torch.Tensor
) -> list[float]:
    """Per anchor, the share of bought candidates (at most ``k``) ranked in the top ``k``."""

    scores = scores.masked_fill(~valid, float("-inf"))
    top = scores.topk(k, dim=1).indices
    hits = bought.gather(1, top).sum(1).float()
    return (hits / bought.sum(1).clamp(max=k).float()).tolist()


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
