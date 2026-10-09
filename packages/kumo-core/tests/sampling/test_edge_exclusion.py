# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import pytest
import torch
from torch_geometric.testing import withCUDA

from kumo_core.features.stores.in_memory_feature import InMemoryFeatureStore
from tests.support.in_memory_graph import InMemoryGraphStore
from tests.support.fixtures import user_store_batch
from kumo_core.sampling.edge_exclusion import drop_edge_index, drop_edge_label_index


@withCUDA
def test_drop_edge_label_index(device: torch.device) -> None:
    edge_label_index = torch.tensor([
        [0, 0, 1, 2, 2],
        [10, 20, 30, 40, 50],
    ], device=device)

    out = drop_edge_label_index(edge_label_index, p=0.0)
    assert out[0].equal(edge_label_index[0])
    assert out[1].equal(edge_label_index[1])
    assert out[2] is None

    out = drop_edge_label_index(edge_label_index, p=0.5)
    assert out[0].size(0) == out[1].size(0)
    assert out[0].size(0) >= 3
    assert out[0].size(0) <= edge_label_index.size(1)
    assert out[0].unique().equal(torch.tensor([0, 1, 2], device=device))
    assert (torch.isin(torch.tensor([10], device=device), out[1]).all()
            or torch.isin(torch.tensor([20], device=device), out[1]).all())
    assert torch.isin(torch.tensor([30], device=device), out[1]).all()
    assert (torch.isin(torch.tensor([40], device=device), out[1]).all()
            or torch.isin(torch.tensor([50], device=device), out[1]).all())
    assert out[2] is not None
    assert out[2].sum() >= 3

    out = drop_edge_label_index(edge_label_index, p=1.0)
    assert out[0].size(0) == out[1].size(0)
    assert out[0].size(0) == 3
    assert out[0].unique().equal(torch.tensor([0, 1, 2], device=device))
    assert (torch.isin(torch.tensor([10], device=device), out[1]).all()
            or torch.isin(torch.tensor([20], device=device), out[1]).all())
    assert torch.isin(torch.tensor([30], device=device), out[1]).all()
    assert (torch.isin(torch.tensor([40], device=device), out[1]).all()
            or torch.isin(torch.tensor([50], device=device), out[1]).all())
    assert out[2] is not None
    assert out[2].sum() == 3


@withCUDA
@pytest.mark.parametrize('num_hops', [0, 1, 2])
def test_drop_edge_index(
    num_hops: int,
    user_store_graph: tuple[
        InMemoryFeatureStore,
        InMemoryGraphStore,
        Any,
    ],
    device: torch.device,
) -> None:
    data = user_store_batch(user_store_graph[0], num_hops).to(device)

    # Message passing edges:
    # USER_ID:   [0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3]
    # STORE_ID': [0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 1, 2]
    # Expected:   -  +  -  -  -  -  +  -  -  -  -  -  -

    edge_label_index = torch.tensor([
        [0, 1, 1, 2, 3, 3, 3],
        [0, 1, 2, 0, 0, 1, 2],
    ], device=device)

    out = drop_edge_index(
        data,
        fact_to_lhs_type=('ORDERS', 'USER_ID', 'USERS'),
        rhs_to_fact_type=('STORES', 'rev_STORE_ID', 'ORDERS'),
        edge_label_index=edge_label_index,
        p=1.0,
    )

    if num_hops < 2:
        assert torch.equal(
            out['ORDERS', 'USER_ID', 'USERS'].edge_index,
            data['ORDERS', 'USER_ID', 'USERS'].edge_index,
        )
        return

    lhs_edge_index = out['ORDERS', 'USER_ID', 'USERS'].edge_index
    assert lhs_edge_index.size() == (2, 2)
    assert lhs_edge_index[0].equal(torch.tensor([1, 6], device=device))
    assert lhs_edge_index[1].equal(torch.tensor([0, 1], device=device))
    assert out['ORDERS', 'USER_ID', 'USERS'].num_sampled_edges == [
        lhs_edge_index.size(1),
        0,
    ]

    out = drop_edge_index(
        data,
        fact_to_lhs_type=('ORDERS', 'USER_ID', 'USERS'),
        rhs_to_fact_type=('STORES', 'rev_STORE_ID', 'ORDERS'),
        edge_label_index=edge_label_index,
        p=0.5,
    )
    lhs_edge_index = out['ORDERS', 'USER_ID', 'USERS'].edge_index
    assert lhs_edge_index.size(0) == 2
    assert lhs_edge_index.size(1) >= 2
    assert lhs_edge_index.size(1) <= 13
    assert out['ORDERS', 'USER_ID', 'USERS'].num_sampled_edges == [
        lhs_edge_index.size(1),
        0,
    ]

    out = drop_edge_index(
        data,
        fact_to_lhs_type=('ORDERS', 'USER_ID', 'USERS'),
        rhs_to_fact_type=('STORES', 'rev_STORE_ID', 'ORDERS'),
        edge_label_index=edge_label_index,
        p=0.0,
    )
    lhs_edge_index = out['ORDERS', 'USER_ID', 'USERS'].edge_index
    assert lhs_edge_index.size(0) == 2
    assert lhs_edge_index.size(1) == 13
    assert out['ORDERS', 'USER_ID', 'USERS'].num_sampled_edges == [
        lhs_edge_index.size(1),
        0,
    ]
