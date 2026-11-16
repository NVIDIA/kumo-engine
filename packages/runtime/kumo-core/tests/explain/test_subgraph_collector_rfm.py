# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from kumoapi.task import TaskType
from torch_geometric.data import HeteroData

from kumo_core.explain.subgraph_collector import (
    get_default_rfm,
    get_unique_counts,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_default_rfm_handles_missing_reverse_edges_on_cuda() -> None:
    batch = HeteroData()
    edge_type = ('USER', 'USER_ID', 'ORDER')
    batch[edge_type].edge_index = torch.tensor(
        [[0, 0], [1, 2]],
        dtype=torch.long,
        device='cuda',
    )
    batch[edge_type].num_sampled_edges = [2]

    rfm = get_default_rfm(
        seed_table='USER',
        task_type=TaskType.BINARY_CLASSIFICATION,
        edge_types=[edge_type],
        batch=batch,
        seed_id=0,
        seed_times=None,
    )

    assert rfm.count == {'ORDER': 2}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_unique_counts_handles_missing_reverse_second_hop_on_cuda() -> None:
    batch = HeteroData()
    first_hop = ('USER', 'USER_ID', 'ORDER')
    batch[first_hop].edge_index = torch.tensor(
        [[0, 0], [1, 2]],
        dtype=torch.long,
        device='cuda',
    )
    batch[first_hop].num_sampled_edges = [2, 0]

    reverse_first_hop = ('ORDER', 'rev_USER_ID', 'USER')
    batch[reverse_first_hop].edge_index = torch.empty(
        (2, 0),
        dtype=torch.long,
        device='cuda',
    )
    batch[reverse_first_hop].num_sampled_edges = [0]

    second_hop = ('ORDER', 'STORE_ID', 'STORE')
    batch[second_hop].edge_index = torch.tensor(
        [[1, 2], [3, 4]],
        dtype=torch.long,
        device='cuda',
    )
    batch[second_hop].num_sampled_edges = [0, 2]

    feature_store = SimpleNamespace(get_pkey_name=lambda table: f'{table}_ID')

    unique_counts = get_unique_counts(
        edge_types=[first_hop],
        batch=batch,
        seed_id=0,
        seed_table='USER',
        feature_store=feature_store,
    )

    assert unique_counts == {'ORDER': {'STORE': 2}}
