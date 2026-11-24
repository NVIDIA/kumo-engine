# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Historical 9a7e9b1 encoder for architecture-cost controls; not a recipe backend."""

from typing import cast

import torch.nn.functional as F
from torch import Tensor, nn

from tfm.compositions.gnn import GNNEncoder
from tfm.contracts.graph import GraphBatch


class LegacyGraphEncoder(nn.Module):
    """Shared GNN implementation. No target head, sequence modules, or table logic.

    Relation-specific mean messages plus a learned self projection per layer.
    Empty neighborhoods retain the self path. Graphs are query-disjoint so
    repeated owners at different cutoffs cannot exchange context.
    """

    def __init__(self, spec: GNNEncoder) -> None:
        super().__init__()
        self.spec = spec
        self.types = tuple(k for k, _ in spec.node_widths)
        self.input = nn.ModuleList(
            nn.Linear(width, spec.hidden_size) for _, width in spec.node_widths
        )
        self.self_layers = nn.ModuleList(
            nn.ModuleList(nn.Linear(spec.hidden_size, spec.hidden_size) for _ in self.types)
            for _ in range(spec.depth)
        )
        self.messages = nn.ModuleList(
            nn.ModuleList(
                nn.Linear(spec.hidden_size, spec.hidden_size, bias=False) for _ in spec.relations
            )
            for _ in range(spec.depth)
        )
        self.norms = nn.ModuleList(
            nn.ModuleList(nn.LayerNorm(spec.hidden_size) for _ in self.types)
            for _ in range(spec.depth)
        )

    @property
    def hidden_size(self) -> int:
        return self.spec.hidden_size

    def forward(self, batch: GraphBatch) -> Tensor:
        if (
            set(batch.node_features) != set(self.types)
            or set(batch.edge_index) != set(self.spec.relations)
            or batch.root_type != self.spec.root_type
        ):
            raise ValueError("Graph batch schema differs from GNN specification")
        if any(batch.node_features[k].shape[1] != width for k, width in self.spec.node_widths):
            raise ValueError("Graph input feature widths differ from GNN specification")
        x = {
            name: F.gelu(layer(batch.node_features[name]))
            for name, layer in zip(self.types, self.input, strict=True)
        }
        for selfs, messages, norms in zip(self.self_layers, self.messages, self.norms, strict=True):
            updated = {
                name: layer(x[name])
                for name, layer in zip(self.types, cast(nn.ModuleList, selfs), strict=True)
            }
            for role, projection in zip(
                self.spec.relations, cast(nn.ModuleList, messages), strict=True
            ):
                source, _, target = role
                send, receive = batch.edge_index[role]
                # Empty roles still participate in autograd. No data-dependent
                # unused message parameters in the eager correctness baseline.
                values = projection(x[source][send])
                sums = values.new_zeros((len(x[target]), self.hidden_size)).index_add(
                    0, receive, values
                )
                counts = values.new_zeros((len(x[target]), 1)).index_add(
                    0, receive, values.new_ones((len(receive), 1))
                )
                updated[target] = updated[target] + sums / counts.clamp_min(1)
            x = {
                name: F.gelu(norm(updated[name]))
                for name, norm in zip(self.types, cast(nn.ModuleList, norms), strict=True)
            }
        return x[batch.root_type][batch.root_index]
