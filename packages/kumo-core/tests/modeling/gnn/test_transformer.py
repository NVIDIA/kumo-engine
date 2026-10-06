# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import warnings
from collections import namedtuple
from dataclasses import is_dataclass

import pytest
import torch
from torch_geometric.testing import onlyFullTest, withCUDA

from kumo_core.modeling.gnn.transformer import GraphTransformer
from kumo_core.modeling.gnn.transformer import TransformerEncoder
from kumo_core.modeling.gnn.positional import positional_encoding_resolver

InputData = namedtuple(
    'InputData',
    [
        'x_dict',
        'edge_index_dict',
        'batch_dict',
        'entity_table_name',
        'num_sampled_nodes_dict',
    ],
)


def get_transformer_model_test_data(batch_size: int, channels: int,
                                    device: torch.device) -> InputData:
    x_dict = {
        "X": torch.randn(batch_size, channels, device=device),
        "Y": torch.randn(200, channels, device=device),
        "Z": torch.randn(300, channels, device=device)
    }
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
    batch_dict = {
        "X": torch.randint(0, batch_size, (batch_size, ), device=device),
        "Y": torch.randint(0, batch_size, (200, ), device=device),
        "Z": torch.randint(0, batch_size, (300, ), device=device)
    }
    entity_table_name = "X"

    return InputData(
        x_dict,
        edge_index_dict,
        batch_dict,
        entity_table_name,
        num_sampled_nodes_dict,
    )


@withCUDA
@pytest.mark.parametrize('channels', [8, 16])
@pytest.mark.parametrize('num_heads', [4, 8])
@pytest.mark.parametrize('num_layers', [1, 2, 3])
@pytest.mark.parametrize('dropout', [0.0, 0.5])
@pytest.mark.parametrize('act', ['relu', 'gelu', None])
@pytest.mark.parametrize('apply_final_norm', [True, False])
def test_transformer_encoder(
    channels: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    act: str | None,
    apply_final_norm: bool,
    device: torch.device,
) -> None:
    if act is None:
        warnings.filterwarnings("ignore", category=UserWarning)

    encoder = TransformerEncoder(
        channels=channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        act=act,
        apply_final_norm=apply_final_norm,
    ).to(device)
    encoder.reset_parameters()

    assert str(encoder) == (f'TransformerEncoder(\n'
                            f'  channels={channels},\n'
                            f'  num_layers={num_layers},\n'
                            f'  num_heads={num_heads},\n'
                            f'  dropout={dropout},\n'
                            f'  act={act},\n'
                            f'  apply_final_norm={apply_final_norm},\n'
                            f')')

    batch, seq_len = 2, 4
    data = torch.rand(batch, seq_len, channels, device=device)
    mask = torch.rand(batch, seq_len, device=device) > 0.5

    out = encoder(data, mask)

    assert out.shape == (batch, seq_len, channels)
    assert out.device == device


def test_transformer_model_config() -> None:
    channels = 8
    batch_size = 100
    seq_len = 32
    test_data = get_transformer_model_test_data(batch_size, channels,
                                                torch.device("cpu"))
    x_dict = test_data.x_dict
    edge_index_dict = test_data.edge_index_dict
    batch_dict = test_data.batch_dict
    entity_table_name = test_data.entity_table_name
    model1 = GraphTransformer(
        channels=channels,
        num_heads=4,
        num_layers=2,
        dropout=0.0,  # crucial to remove randomness in fwd pass
    )
    model1.seq_len = seq_len
    out1 = model1(x_dict, edge_index_dict, batch_dict, batch_size,
                  entity_table_name)

    cfg = model1.config()
    assert is_dataclass(cfg)
    assert cfg._target_ == (
        'kumo_core.modeling.gnn.transformer.GraphTransformer')

    model2 = GraphTransformer.from_config(cfg)
    model2.load_state_dict(model1.state_dict())
    out2 = model2(x_dict, edge_index_dict, batch_dict, batch_size,
                  entity_table_name)

    assert torch.allclose(out1, out2)

    model1.seq_len = 48
    model2.load_state_dict(model1.state_dict())
    assert model2.seq_len == 48


def test_transformer_model_with_positional_encodings_config() -> None:
    channels = 8
    batch_size = 4
    seq_len = 32
    test_data = get_transformer_model_test_data(batch_size, channels,
                                                torch.device("cpu"))
    x_dict = test_data.x_dict
    edge_index_dict = test_data.edge_index_dict
    batch_dict = test_data.batch_dict
    entity_table_name = test_data.entity_table_name
    num_sampled_nodes_dict = test_data.num_sampled_nodes_dict
    model1 = GraphTransformer(
        channels=channels,
        num_heads=4,
        num_layers=2,
        dropout=0.0,  # crucial to remove randomness in fwd pass
        seq_len=seq_len,
        positional_encodings=[
            positional_encoding_resolver(
                "hop",
                channels=channels,
                num_hops=2,
            ),
            positional_encoding_resolver(
                "random_gnn",
                channels=channels,
                num_layers=2,
                edge_types=list(edge_index_dict.keys()),
            )
        ],
    )
    x_dict_1 = x_dict.copy()
    torch.manual_seed(0)
    out1 = model1(x_dict_1, edge_index_dict, batch_dict, batch_size,
                  entity_table_name,
                  num_sampled_nodes_dict=num_sampled_nodes_dict)

    cfg = model1.config()
    assert is_dataclass(cfg)
    assert cfg._target_ == (
        'kumo_core.modeling.gnn.transformer.GraphTransformer')
    assert cfg.positional_encodings[0]._target_ == (
        'kumo_core.modeling.gnn.positional.HopPositionalEncoding')

    model2 = GraphTransformer.from_config(cfg)
    model2.load_state_dict(model1.state_dict())
    torch.manual_seed(0)
    out2 = model2(x_dict, edge_index_dict, batch_dict, batch_size,
                  entity_table_name,
                  num_sampled_nodes_dict=num_sampled_nodes_dict)

    assert torch.allclose(out1, out2)


@withCUDA
@pytest.mark.parametrize('channels', [16, 32])
@pytest.mark.parametrize('num_heads', [4, 8])
@pytest.mark.parametrize('num_layers', [2, 3])
@pytest.mark.parametrize('seq_len', [16, 32])
@pytest.mark.parametrize('dropout', [0.5])
@pytest.mark.parametrize('act', ['relu', 'gelu'])
@pytest.mark.parametrize('apply_final_norm', [True, False])
@pytest.mark.parametrize('positional_encodings_names', [
    [],
    ["random_gnn"],
    ["random_gnn", "hop", "invariant_table_type"],
])
def test_graph_transformer(
    channels: int,
    num_heads: int,
    num_layers: int,
    seq_len: int,
    dropout: float,
    act: str | None,
    apply_final_norm: bool,
    positional_encodings_names: list[str],
    device: torch.device,
) -> None:
    batch_size = 16
    test_data = get_transformer_model_test_data(batch_size, channels, device)
    x_dict = test_data.x_dict
    edge_index_dict = test_data.edge_index_dict
    batch_dict = test_data.batch_dict
    entity_table_name = test_data.entity_table_name
    num_sampled_nodes_dict = test_data.num_sampled_nodes_dict

    positional_encodings = [
        positional_encoding_resolver(
            pos_enc_name,
            channels=channels,
            num_layers=2,
            num_hops=2,
            edge_types=list(edge_index_dict.keys()),
        ) for pos_enc_name in positional_encodings_names
    ]

    model = GraphTransformer(
        channels=channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        act=act,
        apply_final_norm=apply_final_norm,
        seq_len=seq_len,
        positional_encodings=positional_encodings,
    ).to(device)
    model.reset_parameters()
    out = model(x_dict, edge_index_dict, batch_dict, batch_size,
                entity_table_name,
                num_sampled_nodes_dict=num_sampled_nodes_dict)
    assert out.shape == (batch_size, seq_len, channels)
    pos_enc_str = [str(pos_enc) for pos_enc in positional_encodings]
    assert str(model) == (f'GraphTransformer(\n'
                          f'  channels={channels},\n'
                          f'  num_heads={num_heads},\n'
                          f'  num_layers={num_layers},\n'
                          f'  dropout={dropout},\n'
                          f'  act={act},\n'
                          f'  apply_final_norm={apply_final_norm},\n'
                          f'  positional_encodings={pos_enc_str},\n'
                          f')')


@onlyFullTest
@withCUDA
def test_compile(device: torch.device) -> None:
    channels = 8
    batch_size = 10
    seq_len = 32
    test_data = get_transformer_model_test_data(batch_size, channels, device)
    x_dict = test_data.x_dict
    edge_index_dict = test_data.edge_index_dict
    batch_dict = test_data.batch_dict
    entity_table_name = test_data.entity_table_name

    model = GraphTransformer(
        channels=channels,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        seq_len=seq_len,
    ).to(device)

    explanation = torch._dynamo.explain(model)(x_dict, edge_index_dict,
                                               batch_dict, batch_size,
                                               entity_table_name)
    assert explanation.graph_break_count == 0

    compiled_model = torch.compile(model)
    # Note: We call forward/backward twice on purpose. The second call uses
    # 'guards' cached during the first call. The second call is to catch
    # unexpected errors like https://github.com/pytorch/pytorch/issues/111603.
    for _ in range(2):
        out = compiled_model(x_dict, edge_index_dict, batch_dict, batch_size,
                             entity_table_name)
        out.sum().backward(retain_graph=True)


@withCUDA
def test_get_transformer_batch_data_not_all_fit(device: torch.device) -> None:
    """Tests function where entity table has exactly batch_size elements."""
    dims = 2
    batch_size = 3
    entity_table_name = "A"
    batch_dict = {
        entity_table_name: torch.arange(batch_size, device=device),
        "B": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          device=device),
    }
    x_dict = {
        k: torch.randn((len(v), dims), device=device)
        for k, v in batch_dict.items()
    }
    seq_len = 5

    batch_data, mask = GraphTransformer.get_transformer_batch_data(
        x_dict, batch_dict, entity_table_name, batch_size, seq_len)

    assert batch_data.shape == (batch_size, seq_len, dims)
    assert mask.shape == (batch_size, seq_len)

    # first node should be seed node
    assert torch.allclose(batch_data[:, 0], x_dict[entity_table_name])

    # all nodes from B in batch 0 should be there
    assert torch.isin(x_dict["B"][:4], batch_data[0, 1:]).all()
    # 4 nodes from B in batch 1 should be there
    assert torch.isin(x_dict["B"][4:], batch_data[1, 1:]).all(-1).sum() == 4
    # batch 2 should only have the seed node
    assert (batch_data[2, 1:] == 0.).all()
    assert ~mask[2, 0].all()
    assert mask[2, 1:].all()


@withCUDA
def test_get_transformer_batch_data_all_is_batched(
        device: torch.device) -> None:
    """Tests function where many tables have a lot of data but it all fits."""
    dims = 2
    batch = 4
    entity_table_name = "A"
    batch_dict = {
        entity_table_name:
        torch.tensor(list(range(batch)) * 10, device=device),
        "B": torch.tensor(list(range(batch)) * 5, device=device),
        "C": torch.tensor(list(range(batch)) * 15, device=device),
        # nothing for batch 3 in D
        "D": torch.tensor(list(range(batch - 1)) * 30, device=device),
    }
    x_dict = {
        k: torch.randn((len(v), dims), device=device)
        for k, v in batch_dict.items()
    }
    seq_len = 64

    batch_data, mask = GraphTransformer.get_transformer_batch_data(
        x_dict, batch_dict, entity_table_name, batch, seq_len)

    assert batch_data.shape == (batch, seq_len, dims)
    assert mask.shape == (batch, seq_len)

    # first node should be seed node
    assert torch.allclose(batch_data[:, 0], x_dict[entity_table_name][:batch])

    for b in range(batch):
        num_nodes = 30 if b == 3 else 60
        assert (~mask[b]).sum() == num_nodes
        for k in ["B", "C", "D"]:
            is_in_batch = batch_dict[k] == b
            nodes_in_batch = x_dict[k][is_in_batch]
            assert torch.isin(nodes_in_batch, batch_data[b, 1:num_nodes]).all()
        # different for the entity table since some are already seed nodes
        is_in_batch = batch_dict[entity_table_name][batch:] == b
        nodes_in_batch = x_dict[entity_table_name][batch:][is_in_batch]
        assert torch.isin(nodes_in_batch, batch_data[b, 1:num_nodes]).all()


def test_get_seq_len() -> None:
    assert GraphTransformer.get_seq_len(8, 1) == 16
    assert GraphTransformer.get_seq_len(10, 2) == 16
    assert GraphTransformer.get_seq_len(9, 1) == 32
    assert GraphTransformer.get_seq_len(10, 1) == 32
    assert GraphTransformer.get_seq_len(30, 2) == 32
    assert GraphTransformer.get_seq_len(150, 10) == 32
    assert GraphTransformer.get_seq_len(300, 10) == 64
    assert GraphTransformer.get_seq_len(31, 1) == 64
    assert GraphTransformer.get_seq_len(32, 1) == 64
    assert GraphTransformer.get_seq_len(33, 1) == 128
    assert GraphTransformer.get_seq_len(600, 10) == 128
    assert GraphTransformer.get_seq_len(1000, 10) == 256
    assert GraphTransformer.get_seq_len(10000, 10) == 256
    assert GraphTransformer.get_seq_len(100000, 10) == 256
