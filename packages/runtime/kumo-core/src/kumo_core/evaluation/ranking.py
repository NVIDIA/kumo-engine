# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import torchmetrics
from torch import Tensor


class MulticlassMRR(torchmetrics.Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self) -> None:
        super().__init__()

        self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')

    def update(self, pred: Tensor, target: Tensor) -> None:
        argsort = pred.argsort(dim=-1, descending=True)
        match = (argsort == target.view(-1, 1)).to(self.accum.dtype)
        rank = match.argmax(dim=-1).to(self.accum.dtype) + 1
        reciprocal_rank = 1.0 / rank
        reciprocal_rank[match.sum(dim=-1) == 0.0] = 0.0

        self.accum += reciprocal_rank.sum()
        self.total += pred.size(0)

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.zeros_like(self.accum)
        return self.accum / self.total
