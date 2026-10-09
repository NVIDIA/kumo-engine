# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo.testing.graph import get_random_edge_index


def test_get_random_edge_index():
    num_src_nodes = 10
    num_dst_nodes = 20
    num_edges = 100
    edge_index = get_random_edge_index(num_src_nodes, num_dst_nodes, num_edges)
    assert (edge_index[0] < num_src_nodes).all()
    assert (edge_index[1] < num_dst_nodes).all()
    assert (edge_index >= 0).all()
    assert edge_index.shape == (2, num_edges)
