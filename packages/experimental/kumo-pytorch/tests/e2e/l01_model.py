# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combined graph, history and aggregate consumer for the L01 training benchmark."""

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, PositionalEncoding, SAGEConv

from e2e.datasets.aml import _AMOUNTS, _CATEGORIES, _ROLES, l01_loss
from kumo.pytorch.models.causal import CausalEncoderConfig, CausalTokenEncoder
from tests.reference_models.batch_adapters import nuformer_batch


def _numerical(fields, names):
    values = torch.stack([fields[name]["values"] for name in names], dim=1)
    valid = torch.stack([fields[name]["valid"] for name in names], dim=1)
    return torch.cat((values.masked_fill(~valid, 0), valid), dim=1)


class _Combined(nn.Module):
    def __init__(self, task):
        super().__init__()
        space = task.spaces["history"]
        self.history = CausalTokenEncoder(
            CausalEncoderConfig(d_model=16, d_ffn=64, attention_heads=2, layers=1, dropout=0),
            vocabulary_size=space.size,
            pad_id=space.special(role="pad"),
        )
        fields = task.output_schema["inputs"]["graph"]["transactions"]
        self.amounts = nn.Linear(2 * len(_AMOUNTS), 16)
        self.categories = nn.ModuleDict(
            {name: nn.Embedding(fields[name]["vocabulary_size"] + 1, 16) for name in _CATEGORIES}
        )
        self.time = nn.Sequential(PositionalEncoding(16), nn.Linear(16, 16))
        self.account = nn.Parameter(torch.zeros(1, 16))
        edges = [("accounts", role, "transactions") for role in _ROLES]
        edges += [("transactions", role, "accounts") for role in _ROLES]
        self.layers = nn.ModuleList(
            HeteroConv({edge: SAGEConv(16, 16) for edge in edges}) for _ in range(2)
        )
        # Only account embeddings feed the prediction after the second message pass.
        for role in _ROLES:
            del self.layers[-1].convs["accounts", role, "transactions"]
        self.norms = nn.ModuleList(
            nn.ModuleDict({name: nn.LayerNorm(16) for name in destinations})
            for destinations in (("accounts", "transactions"), ("accounts",))
        )
        self.counts = tuple(
            f.field for f in task.contract.batch.inputs["aggregates"].representation.fields
        )
        self.aggregates = nn.Sequential(
            nn.Linear(2 * len(self.counts), 8), nn.GELU(), nn.Linear(8, 8)
        )
        currencies = task.output_schema["targets"]["currency"]["value"]["vocabulary_size"]
        self.head = nn.Sequential(nn.Linear(40, 16), nn.GELU(), nn.Linear(16, 1 + currencies))

    def forward(self, batch):
        graph = batch.inputs["graph"]
        accounts, transactions = (graph["nodes"][name] for name in ("accounts", "transactions"))
        fields = transactions["features"]
        tx = self.amounts(_numerical(fields, _AMOUNTS))
        for name, embedding in self.categories.items():
            field = fields[name]
            tx = tx + embedding(
                field["values"].masked_fill(~field["valid"], embedding.num_embeddings - 1)
            )
        # L01 samples the graph at T-1 second and measures transaction ages at prediction T.
        clock = batch.metadata["as_of_time"][transactions["batch"]] + 1
        tx = tx + self.time((clock - fields["event_time_seconds"]["values"]) / 86400)
        nodes = {"accounts": self.account.expand(len(accounts["ids"]), -1), "transactions": tx}
        edges = {}
        for role in _ROLES:
            index = graph["edges"][role]["index"]
            edges["accounts", role, "transactions"] = index
            edges["transactions", role, "accounts"] = index.flip(0)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            nodes = {name: norm[name](value).relu() for name, value in layer(nodes, edges).items()}
        owners = accounts["batch"]
        roots = accounts["ids"] == batch.metadata["entity_ids"][owners]
        account = tx.new_zeros((batch.batch_size, 16))
        account.index_add_(0, owners, nodes["accounts"] * roots[:, None])
        sequence = nuformer_batch(batch)
        history = (
            self.history(sequence).representations["entity"]
            if sequence.input_ids.size(1)
            else torch.zeros_like(account)
        )
        aggregates = self.aggregates(_numerical(batch.inputs["aggregates"], self.counts))
        return self.head(torch.cat((account, history, aggregates), dim=1))


def make_l01_step(task, device):
    """Initialize the model and optimizer once for both epochs of this worker."""
    model = _Combined(task).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)

    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, _, counts = l01_loss(batch, outputs[:, 0], outputs[:, 1:])
        torch._assert_async(loss.isfinite(), "Nonfinite L01 loss")
        update = any(counts.values())
        if update:
            loss.backward()
            optimizer.step()
        return counts["amount"], counts["currency"], update

    return step
