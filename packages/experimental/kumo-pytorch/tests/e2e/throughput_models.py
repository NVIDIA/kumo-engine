# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model and loss for each manual throughput workload."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from kumo.pytorch.models.causal import CausalEncoderConfig, CausalTokenEncoder
from tests.e2e.l01_model import L01Model
from tests.reference_models.batch_adapters import nuformer_batch


class CountSAGE(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph = SAGEConv((2, 2), 16, root_weight=False)
        self.count = nn.Linear(16, 1)

    def forward(self, batch):
        graph = batch.inputs["graph"]
        clients, transactions = (graph["nodes"][table] for table in ("roots", "transactions"))
        root, event = graph["edges"]["client_transactions"]["index"]
        features = transactions["features"].values()
        values = torch.stack(
            [torch.asinh(f["values"].float()).masked_fill(~f["valid"], 0) for f in features], dim=1
        )
        valid = torch.stack([f["valid"] for f in features], dim=1).float()
        x = torch.stack((values.mean(1), valid.mean(1)), dim=1)
        edges = torch.stack((event, clients["batch"][root]))
        pooled = self.graph((x, None), edges, size=(len(x), batch.batch_size))
        return self.count(pooled).squeeze(1)

    def batch_loss(self, batch, device):
        if not batch.targets["count"]["valid"].any():
            return None
        batch = batch.to(device, non_blocking=True)
        target = batch.targets["count"]
        valid = target["valid"]
        return F.smooth_l1_loss(self(batch)[valid], target["values"][valid])


class NextTokenModel(nn.Module):
    def __init__(self, space):
        super().__init__()
        config = CausalEncoderConfig(d_model=16, d_ffn=32, attention_heads=2, layers=1, dropout=0)
        self.sequence = CausalTokenEncoder(
            config, vocabulary_size=space.size, pad_id=space.special(role="pad")
        )

    def batch_loss(self, batch, device):
        valid = batch.targets["next_token"]["valid"]
        if not valid.any():
            return None
        sequence = nuformer_batch(batch).to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        hidden = self.sequence(sequence).contextual
        logits = hidden[valid] @ self.sequence.embedding.weight.T
        return F.cross_entropy(logits, sequence.target_ids[valid])


def make_model(workload, task):
    if workload == "M04":
        return CountSAGE()
    if workload == "S02":
        return NextTokenModel(task.spaces["history"])
    if workload == "L01":
        return L01Model(task)
    raise ValueError(f"Unknown throughput workload: {workload}")
