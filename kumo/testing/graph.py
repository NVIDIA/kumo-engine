# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


def get_random_edge_index(
    num_src_nodes: int,
    num_dst_nodes: int,
    num_edges: int,
) -> Tensor:
    r"""Randomly generates an edge_index with specified information."""
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)
