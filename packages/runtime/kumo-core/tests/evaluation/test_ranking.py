# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch_geometric.testing import withCUDA

from kumo_core.evaluation.ranking import MulticlassMRR


@withCUDA
def test_mrr(device: torch.device) -> None:
    pred = torch.tensor([
        [0.4, 0.2, 0.1, 0.3],
        [0.1, 0.2, 0.3, 0.4],
        [0.3, 0.4, 0.2, 0.1],
    ], device=device)
    target = torch.tensor([3, 0, 1], device=device)

    metric = MulticlassMRR().to(device)
    assert float(metric.compute()) == 0.0  # type: ignore

    for _ in range(2):
        metric.update(pred, target)
        out = float(metric.compute())  # type: ignore
        assert out == pytest.approx((0.5 + 0.25 + 1) / 3)

    metric.reset()
    assert float(metric.compute()) == 0.0  # type: ignore


@withCUDA
def test_mrr_out_of_bounds(device: torch.device) -> None:
    pred = torch.tensor([
        [0.4, 0.2, 0.1, 0.3],
        [0.1, 0.2, 0.3, 0.4],
    ], device=device)
    target = torch.tensor([4, 3], device=device)

    metric = MulticlassMRR().to(device)
    metric.update(pred, target)
    assert float(metric.compute()) == 0.5  # type: ignore
