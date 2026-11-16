# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Iterable, List

import torch
from torch import Tensor
from torch_geometric.utils import sort_edge_index


def parse_edge_index(edge_index: Any) -> Tensor:
    if isinstance(edge_index, tuple) and len(edge_index) == 2:
        return torch.stack(edge_index, dim=0)
    if isinstance(edge_index, list):
        return torch.tensor(edge_index)
    return edge_index


def assert_edge_index_equal(edge_index1: Any, edge_index2: Any):
    r"""Tests whether two :obj:`edge_index` tensors are equivalent."""
    edge_index1 = parse_edge_index(edge_index1)
    edge_index2 = parse_edge_index(edge_index2)
    assert torch.equal(sort_edge_index(edge_index1),
                       sort_edge_index(edge_index2))


def compare_nested_lists(expected_target: List[Any], target: Iterable):
    for exp_t, target_item in zip(expected_target, target):
        if isinstance(target_item, str):
            t = json.loads(target_item)
        else:
            t = list(target_item)
        assert len(t) == len(exp_t)
        for a, b in zip(exp_t, t):
            assert a == b
