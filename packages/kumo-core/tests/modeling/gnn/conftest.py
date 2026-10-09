# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

import pytest
import torch
from torch_geometric.data import HeteroData


def _get_mini_batch(
    batch_size: int = 1024,
    num_targets: int = 64,
    channels: int = 64,
    device: torch.device | None = None,
) -> HeteroData:
    r"""Returns a (fake) mini-batch of data, which models the basic use-case of
    a :obj:`USER->TRANS->STORE` scenario with 2-hop sampling.

    Args:
        batch_size (int, optional): The batch size/the number of entities in
            the batch. (default: :obj:`1024`)
        num_targets (int, optional): The number of target nodes that were
            sampled for each entity. (default: :obj:`64`)
        channels (int, optional): The number of channels of all node features.
            (default: :obj:`64`)
        device (torch.device, optional): The device to store the data on.
            (default: :obj:`None`)
    """
    num_users = batch_size
    num_trans = num_users * num_targets
    num_stores = num_trans // 2  # Approximate number of stores per batch.

    batch = HeteroData()
    batch['user'].x = torch.randn(num_users, channels, device=device)
    batch['user'].num_sampled_nodes = [num_users, 0, 0]

    batch['trans'].x = torch.randn(num_trans, channels, device=device)
    batch['trans'].num_sampled_nodes = [0, num_trans, 0]

    batch['store'].x = torch.randn(num_stores, channels, device=device)
    batch['store'].num_sampled_nodes = [0, 0, num_stores]

    # many-to-one mapping between stores and transactions (single direction):
    edge_index = torch.stack([
        torch.arange(num_trans, device=device) % num_stores,
        torch.arange(num_trans, device=device),
    ], dim=0)
    batch['store', 'rev_to', 'trans'].edge_index = edge_index
    batch['store', 'rev_to', 'trans'].num_sampled_edges = [0, num_trans]

    batch['trans', 'to', 'store'].edge_index = edge_index[:, :0]
    batch['trans', 'to', 'store'].num_sampled_edges = [0, 0]

    # many-to-one mapping between users and transactions (bidirectional):
    edge_index = torch.stack([
        torch.arange(num_users, device=device).view(-1, 1).repeat(
            1, num_targets).view(-1),
        torch.arange(num_trans, device=device),
    ], dim=0)
    batch['user', 'rev_to', 'trans'].edge_index = edge_index
    batch['user', 'rev_to', 'trans'].num_sampled_edges = [0, num_trans]

    batch['trans', 'to', 'user'].edge_index = edge_index.flip([0])
    batch['trans', 'to', 'user'].num_sampled_edges = [num_trans, 0]

    return batch


@pytest.fixture()
def get_mini_batch() -> Callable:
    return _get_mini_batch
