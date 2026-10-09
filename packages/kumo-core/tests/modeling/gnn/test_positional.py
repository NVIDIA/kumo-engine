# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import Tensor
from torch_geometric.testing import withCUDA
from torch_geometric.typing import EdgeType, NodeType

from kumo_core.modeling.gnn.positional import (
    HopPositionalEncoding,
    InvariantRandomGNNPositionalEncoding,
    InvariantTableTypePositionalEncoding,
    RandomGNNPositionalEncoding,
    positional_encoding_resolver,
)


def get_test_data(
    batch_size: int, device: torch.device
) -> tuple[dict[EdgeType, Tensor], dict[NodeType, list[int]]]:
    num_sampled_nodes_dict = {
        "X": [batch_size, 0, 0],
        "Y": [0, 200, 0],
        "Z": [0, 0, 300]
    }
    edge_index_dict = {
        ("X", "e1", "Y"): torch.randint(0, batch_size, (2, 200),
                                        device=device),
        ("Y", "rev_e1", "X"): torch.randint(0, batch_size, (2, 200),
                                            device=device),
        ("Y", "e2", "Z"): torch.randint(0, 200, (2, 300), device=device),
    }
    return edge_index_dict, num_sampled_nodes_dict


@withCUDA
@pytest.mark.parametrize('channels', [4, 8])
@pytest.mark.parametrize('num_layers', [2, 3])
def test_random_gnn_positional_embedder(
    channels: int,
    num_layers: int,
    device: torch.device,
) -> None:
    edge_index_dict, num_sampled_nodes_dict = get_test_data(100, device)
    edge_types = list(edge_index_dict.keys())

    pos_gnn = RandomGNNPositionalEncoding(
        channels=channels,
        num_layers=num_layers,
        edge_types=edge_types,
    ).to(device)
    output = pos_gnn(edge_index_dict, num_sampled_nodes_dict)

    assert isinstance(output, dict)
    assert set(output.keys()) == set(num_sampled_nodes_dict.keys())

    for key in output:
        assert output[key].shape == (sum(num_sampled_nodes_dict[key]),
                                     channels)
        assert torch.isfinite(output[key]).all()


@withCUDA
@pytest.mark.parametrize('channels', [4, 8])
@pytest.mark.parametrize('num_layers', [2, 3])
def test_invariant_random_gnn_pe(
    channels: int,
    num_layers: int,
    device: torch.device,
) -> None:
    edge_index_dict, num_sampled_nodes_dict = get_test_data(100, device)

    pe = InvariantRandomGNNPositionalEncoding(
        channels=channels,
        num_layers=num_layers,
    ).to(device)
    pe.reset_parameters()
    assert str(pe) == 'InvariantRandomGNNPositionalEncoding()'

    output = pe(edge_index_dict, num_sampled_nodes_dict)

    assert isinstance(output, dict)
    assert set(output.keys()) == set(num_sampled_nodes_dict.keys())

    for key in output:
        assert output[key].size() == (
            sum(num_sampled_nodes_dict[key]),
            channels,
        )
        assert torch.isfinite(output[key]).all()


@withCUDA
@pytest.mark.parametrize('channels', [2, 4])
@pytest.mark.parametrize('num_hops', [2, 3])
def test_hop_positional_embedder(
    channels: int,
    num_hops: int,
    device: torch.device,
) -> None:
    edge_index_dict, num_sampled_nodes_dict = get_test_data(100, device)
    if num_hops != 2:
        num_hop_indices = num_hops + 1
        for k, v in num_sampled_nodes_dict.items():
            num_nodes = sum(v)
            sampled_nodes = [num_nodes // num_hop_indices] * num_hop_indices
            remaining_nodes = num_nodes - sum(sampled_nodes)
            sampled_nodes[0] += remaining_nodes
            num_sampled_nodes_dict[k] = sampled_nodes

    pos_hop = HopPositionalEncoding(
        channels=channels,
        num_hops=num_hops,
    ).to(device)

    output = pos_hop(edge_index_dict, num_sampled_nodes_dict)

    assert isinstance(output, dict)
    assert set(output.keys()) == set(num_sampled_nodes_dict.keys())

    for table_name, pos_embeds in output.items():
        assert pos_embeds.shape == (sum(num_sampled_nodes_dict[table_name]),
                                    channels)
        assert torch.isfinite(pos_embeds).all()

        # Check that the hop embeddings actually match the expected values
        limits = torch.cumsum(
            torch.tensor([0] + num_sampled_nodes_dict[table_name],
                         device=device, dtype=torch.long), dim=0)
        for i in range(1, limits.shape[0]):
            this_hop = pos_embeds[limits[i - 1]:limits[i]]
            expected_hop = pos_hop.emb(
                torch.tensor(i - 1, device=device, dtype=torch.long))
            expected_hop = expected_hop.unsqueeze(0)
            assert torch.allclose(this_hop, expected_hop)


@withCUDA
def test_table_type_positional_encoding_forward(device: torch.device) -> None:
    channels = 8
    pos_enc = InvariantTableTypePositionalEncoding(
        channels=channels).to(device)

    for keys in [["A", "B", "C"], ["D", "E"], ["F", "G", "A", "B"]]:
        num_sampled_nodes_dict = {k: [10, 0, 4] for k in keys}

        output = pos_enc({}, num_sampled_nodes_dict)

        assert isinstance(output, dict)
        assert set(output.keys()) == set(num_sampled_nodes_dict.keys())
        for value in output.values():
            assert value.shape == (14, channels)
            assert torch.isfinite(value).all()


def test_positional_embedding_resolver() -> None:
    channels = 8
    num_layers = 2
    num_hops = 2
    edge_types = [("X", "e1", "Y"), ("Y", "rev_e1", "X"), ("Y", "e2", "Z")]

    pos_embedder = positional_encoding_resolver(
        "random_gnn",
        channels=channels,
        num_layers=num_layers,
        num_hops=num_hops,
        edge_types=edge_types,
    )
    assert isinstance(pos_embedder, RandomGNNPositionalEncoding)

    pos_embedder = positional_encoding_resolver(
        "hop",
        channels=channels,
        num_layers=num_layers,
        num_hops=num_hops,
        edge_types=edge_types,
    )
    assert isinstance(pos_embedder, HopPositionalEncoding)

    pos_embedder = positional_encoding_resolver(
        "invariant_table_type",
        channels=channels,
        num_layers=num_layers,
        num_hops=num_hops,
        edge_types=edge_types,
    )
    assert isinstance(pos_embedder, InvariantTableTypePositionalEncoding)
