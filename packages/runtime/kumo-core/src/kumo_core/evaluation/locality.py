# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import torchmetrics
from torch import Tensor
from torch_geometric.utils import scatter


class LocalityScore(torchmetrics.Metric):
    r"""Computes the locality score as defined in
    `"ContextGNN: Beyond Two-Tower Recommendation Systems"
    <https://arxiv.org/abs/2411.19513>`_, *i.e.*, it measures the fraction of
    ground-truth links that are part of the sampled RHS node set.
    """
    is_differentiable: bool = False
    full_state_update: bool = False
    higher_is_better: bool | None = None

    def __init__(self) -> None:
        super().__init__()

        self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')

    def update(
        self,
        edge_label_index: Tensor | tuple[Tensor, Tensor],
        rhs_node: Tensor,
        rhs_batch: Tensor,
    ) -> None:
        r"""Updates the state variables based on the current mini-batch
        and its ground-truth.

        :meth:`update` can be repeated multiple times to accumulate the results
        of successive predictions, *e.g.*, inside a mini-batch training or
        evaluation loop.

        Args:
            edge_label_index: The ground-truth indices for every example in the
                mini-batch, given in COO format of shape
                :obj:`[2, num_ground_truth_indices]`.
            rhs_node: The global node indices of RHS nodes in the sampled
                subgraph.
            rhs_batch: The batch identifier for every RHS node in the sampled
                subgraph.
        """
        y_batch, y = edge_label_index

        batch_size = max(
            int(y_batch.max()) + 1 if y_batch.numel() > 0 else 0,
            int(rhs_batch.max()) + 1 if rhs_batch.numel() > 0 else 0,
        )

        y_idx = y * batch_size + y_batch
        rhs_idx = rhs_node * batch_size + rhs_batch

        # How many ground-truth links are part of RHS sampled node set:
        mask = torch.isin(y_idx, rhs_idx).to(self.accum.dtype)
        score = scatter(mask, y_batch, dim_size=batch_size, reduce='mean')
        ones = self.total.new_ones(y_batch.size())
        count = scatter(ones, y_batch, dim_size=batch_size, reduce='sum')

        self.accum += score.sum()
        self.total += (count > 0).sum()

    def compute(self) -> Tensor:
        r"""Computes the final locality score."""
        if self.total == 0:
            return torch.zeros_like(self.accum)
        return self.accum / self.total

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
