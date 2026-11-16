# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType
from torch_geometric.utils import cumsum, scatter


def drop_edge_label_index(
    edge_label_index: Tensor | tuple[Tensor, Tensor],
    p: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    r"""Drops ground-truth labels according to a given probability.

    For every example, at least one ground-truth label is preserved.

    Args:
        edge_label_index: The ground-truth indices for every example in the
            mini-batch, given in COO format of shape
            :obj:`[2, num_ground_truth_indices]`.
        p: Probability of a label to be dropped.
        generator: A pseudo-random number generator for sampling.
    """
    assert 0.0 <= p <= 1.0

    if p == 0.0:  # Nothing to do.
        return edge_label_index[0], edge_label_index[1], None

    batch, y = edge_label_index
    label_mask = torch.rand(
        y.size(0),
        generator=generator,
        device=y.device,
    ) >= p

    # Select a single random edge per entity:
    deg = scatter(torch.ones_like(batch), batch, dim=0)
    prob = torch.rand(deg.size(0), generator=generator, device=y.device)
    node_id = (prob * deg).long() + cumsum(deg)[:-1]

    # Preserve at least one ground-truth label:
    node_mask = (scatter(label_mask.float(), batch, dim=0) == 0) & (deg > 0)
    label_mask[node_id[node_mask]] = True

    return batch[label_mask], y[label_mask], label_mask


def drop_edge_index(
    data: HeteroData,
    fact_to_lhs_type: EdgeType,
    rhs_to_fact_type: EdgeType,
    edge_label_index: Tensor | tuple[Tensor, Tensor],
    p: float,
    num_rhs_nodes: int | None = None,
    generator: torch.Generator | None = None,
) -> HeteroData:
    r"""Drops ground-truth links from the subgraph of a
    :class:`~torch_geometric.data.HeteroData` object according to a given
    probability.

    Args:
        data: The heterogeneous graph.
        fact_to_lhs_type: The fact to LHS edge type.
        rhs_to_fact_type: The RHS to fact edge type.
        edge_label_index: The ground-truth indices for every example in the
            mini-batch, given in COO format of shape
            :obj:`[2, num_ground_truth_indices]`.
        p: Probability of a label to be excluded from message passing.
        num_rhs_nodes: The number of RHS nodes.
        generator: A pseudo-random number generator for sampling.
    """
    assert 0.0 <= p <= 1.0

    if p == 0.0:  # Nothing to do.
        return data

    # Subsample `edge_label_index` based on dropout ratio:
    batch, y = edge_label_index
    label_mask = torch.rand(
        y.size(0),
        generator=generator,
        device=y.device,
    ) < p
    batch, y = batch[label_mask], y[label_mask]
    if batch.numel() == 0:  # Nothing to do.
        return data

    # Collect LHS facts:
    lhs_edge_index = data[fact_to_lhs_type].edge_index
    if lhs_edge_index.numel() == 0:  # < 1-hop sampling.
        return data
    lhs_num_sampled_edges = data[fact_to_lhs_type].get('num_sampled_edges')
    if lhs_num_sampled_edges is not None:
        lhs_edge_index = lhs_edge_index[:, :lhs_num_sampled_edges[0]]
    if lhs_edge_index.numel() == 0:  # < 1-hop sampling.
        return data
    lhs_fact, lhs = lhs_edge_index

    # Find the corresponding RHS IDs that belong to each LHS fact:
    rhs_edge_index = data[rhs_to_fact_type].edge_index
    if rhs_edge_index.numel() == 0:  # < 2-hop sampling.
        return data
    rhs_num_sampled_edges = data[rhs_to_fact_type].get('num_sampled_edges')
    if rhs_num_sampled_edges is not None:
        rhs_edge_index = rhs_edge_index[:, rhs_num_sampled_edges[0]:]
        rhs_edge_index = rhs_edge_index[:, :rhs_num_sampled_edges[1]]
    if rhs_edge_index.numel() == 0:  # < 2-hop sampling.
        return data
    rhs, rhs_fact = rhs_edge_index

    rhs_n_id = rhs.new_full((rhs_fact.max() + 1, ), fill_value=-1)
    rhs_n_id[rhs_fact] = data[rhs_to_fact_type[0]].n_id[rhs]
    rhs_n_id = rhs_n_id[lhs_fact]  # May contain -1 in case of dangling fkeys.

    # Find overlap between labels and message passing edges:
    if num_rhs_nodes is None:
        num_rhs_nodes = max(int(y.max()), int(rhs_n_id.max()))

    label_id = batch * num_rhs_nodes + y
    edge_id = lhs * num_rhs_nodes + rhs_n_id
    edge_id[rhs_n_id == -1] = -1

    dropout_edge_mask = ~torch.isin(edge_id, label_id)

    # Filter out message passing edges:
    data = copy.copy(data)
    lhs_edge_index = lhs_edge_index[:, dropout_edge_mask]
    if lhs_num_sampled_edges is not None:
        new_num_first_hop_edges = lhs_edge_index.size(1)
        lhs_edge_index = torch.cat([
            lhs_edge_index,
            data[fact_to_lhs_type].edge_index[:, lhs_num_sampled_edges[0]:],
        ], dim=1)
        lhs_num_sampled_edges = ([new_num_first_hop_edges] +
                                 lhs_num_sampled_edges[1:])
        data[fact_to_lhs_type].num_sampled_edges = lhs_num_sampled_edges
    data[fact_to_lhs_type].edge_index = lhs_edge_index

    return data
