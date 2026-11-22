# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU training for the weekly session-count task."""

import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from kumo.pytorch import api


class CountSAGE(nn.Module):
    def __init__(self, task: api.Task):
        super().__init__()
        fields = task.output_schema["inputs"]["graph"]["sessions"]
        self.device_embedding = nn.Embedding(fields["device_type"]["vocabulary_size"], 4)
        # Viewers have no features. Aggregate messages from their historical sessions.
        self.conv = SAGEConv(6, 16, root_weight=False)
        self.count = nn.Linear(16, 1)

    def forward(self, batch: api.Batch) -> torch.Tensor:
        graph = batch.inputs["graph"]
        fields = graph["nodes"]["sessions"]["features"]
        numeric = torch.stack(
            [fields[key]["values"] for key in ("watch_seconds", "completion_frac")], 1
        )
        sessions = torch.cat((numeric, self.device_embedding(fields["device_type"]["values"])), 1)
        viewer, session = graph["edges"]["viewers_sessions"]["index"]
        owners = graph["nodes"]["viewers"]["batch"][viewer]
        neighbors = self.conv(
            (sessions, None), torch.stack((session, owners)), size=(len(sessions), batch.batch_size)
        )
        return self.count(neighbors.relu()).squeeze(-1)

    def loss(self, batch):
        target = batch.targets["count"]
        predicted = self(batch)
        assert target["valid"].all() and not any(batch.metadata["is_padding"])
        assert predicted.shape == target["values"].shape == (batch.batch_size,)
        assert predicted.dtype == target["values"].dtype == torch.float32
        loss = nn.functional.smooth_l1_loss(predicted.double(), target["values"].double())
        assert torch.isfinite(loss)
        return loss
