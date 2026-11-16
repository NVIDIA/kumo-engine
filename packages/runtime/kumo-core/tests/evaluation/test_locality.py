# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch_geometric.testing import withCUDA

from kumo_core.evaluation.locality import LocalityScore


@withCUDA
def test_locality_score(device: torch.device) -> None:
    metric = LocalityScore().to(device)
    assert str(metric) == 'LocalityScore()'

    edge_label_index = torch.tensor([
        [0, 0, 1, 1, 1, 3],
        [2, 3, 0, 1, 2, 3],
    ], device=device)

    rhs_node = torch.tensor([1, 2, 0, 3], device=device)
    rhs_batch = torch.tensor([0, 0, 3, 3], device=device)

    assert float(metric.compute()) == 0.0  # type: ignore
    expected = torch.tensor((1 / 2 + 1) / 3, device=device)
    metric.update(edge_label_index, rhs_node, rhs_batch)
    assert metric.compute().allclose(expected)  # type: ignore
    metric.update(edge_label_index, rhs_node, rhs_batch)
    assert metric.compute().allclose(expected)  # type: ignore
    metric.reset()
    assert float(metric.compute()) == 0.0  # type: ignore


@withCUDA
def test_empty(device: torch.device) -> None:
    metric = LocalityScore().to(device)

    metric.update(  # type: ignore
        edge_label_index=torch.empty(2, 0, device=device),
        rhs_node=torch.empty(0, device=device),
        rhs_batch=torch.empty(0, device=device),
    )
    assert float(metric.compute()) == 0.0  # type: ignore
