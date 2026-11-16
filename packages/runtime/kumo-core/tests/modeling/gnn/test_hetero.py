# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import is_dataclass

import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.testing import onlyFullTest, withCUDA
from torch_geometric.typing import WITH_PT24

from kumo_core.modeling.gnn.hetero import HeteroGNN


@withCUDA
@pytest.mark.parametrize('num_pre_mp_layers', [0, 2])
@pytest.mark.parametrize('aggr', ['sum', ['sum', 'mean', 'max']])
@pytest.mark.parametrize('dropout', [0.0, 0.5])
@pytest.mark.parametrize('act', [None, 'relu', 'leaky_relu', 'elu', 'gelu'])
@pytest.mark.parametrize('act_first', [False, True])
@pytest.mark.parametrize('norm', [None, 'layer_norm', 'batch_norm'])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('learn_skip', [False, True])
def test_hetero_gnn_basic(
    num_pre_mp_layers: int,
    aggr: str | list[str],
    dropout: float,
    act: str | None,
    act_first: bool,
    norm: str | None,
    bias: bool,
    learn_skip: bool,
    device: torch.device,
) -> None:
    data = HeteroData()
    data['v1'].x = torch.randn(100, 8, device=device)
    data['v2'].x = torch.randn(100, 8, device=device)
    edge_index = torch.randint(0, 100, (2, 500), device=device)
    data['v1', 'to', 'v2'].edge_index = edge_index
    data['v2', 'rev_to', 'v1'].edge_index = edge_index.flip([0])

    model = HeteroGNN(
        channels=8,
        num_mp_layers=2,
        metadata=data.metadata(),
        num_pre_mp_layers=num_pre_mp_layers,
        aggr=aggr,
        dropout=dropout,
        act=act,
        act_first=act_first,
        norm=norm,
        bias=bias,
        learn_skip=learn_skip,
        device=device,
    )
    model.reset_parameters()

    assert str(model) == (f"HeteroGNN(\n"
                          f"  channels=8,\n"
                          f"  num_mp_layers=2,\n"
                          f"  num_pre_mp_layers={num_pre_mp_layers},\n"
                          f"  aggr={aggr},\n"
                          f"  dropout={dropout},\n"
                          f"  act={act},\n"
                          f"  act_first={act_first},\n"
                          f"  norm={norm},\n"
                          f"  bias={bias},\n"
                          f"  learn_skip={learn_skip},\n"
                          f")")

    out_dict = model(data.x_dict, data.edge_index_dict)
    assert len(out_dict) == 2
    assert out_dict['v1'].device == device
    assert out_dict['v1'].size() == (100, 8)
    assert out_dict['v2'].device == device
    assert out_dict['v2'].size() == (100, 8)

    # Test empty edge indices:
    edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    data['v1', 'to', 'v2'].edge_index = edge_index
    data['v2', 'rev_to', 'v1'].edge_index = edge_index.flip([0])

    out_dict = model(data.x_dict, data.edge_index_dict)
    assert len(out_dict) == 2
    assert out_dict['v1'].device == device
    assert out_dict['v1'].size() == (100, 8)
    assert out_dict['v2'].device == device
    assert out_dict['v2'].size() == (100, 8)


def test_hetero_gnn_config() -> None:
    data = HeteroData()
    data['v1'].x = torch.randn(100, 8)
    data['v2'].x = torch.randn(100, 8)
    edge_index = torch.randint(0, 100, (2, 500))
    data['v1', 'v2'].edge_index = edge_index
    data['v2', 'v1'].edge_index = edge_index.flip([0])

    model1 = HeteroGNN(
        channels=8,
        num_mp_layers=2,
        metadata=data.metadata(),
        num_pre_mp_layers=1,
        aggr=['sum', 'mean'],
        dropout=0.0,
        act='elu',
        act_first=True,
        norm='layer_norm',
        bias=False,
        learn_skip=False,
    )
    out_dict1 = model1(data.x_dict, data.edge_index_dict)

    cfg = model1.config()
    assert is_dataclass(cfg)
    assert cfg._target_ == 'kumo_core.modeling.gnn.hetero.HeteroGNN'

    model2 = HeteroGNN.from_config(cfg)
    model2.load_state_dict(model1.state_dict())
    out_dict2 = model2(data.x_dict, data.edge_index_dict)

    for out1, out2 in zip(out_dict1.values(), out_dict2.values()):
        assert torch.allclose(out1, out2)


@withCUDA
@pytest.mark.parametrize('batch_size', [64])
@pytest.mark.parametrize('channels', [16])
def test_patch_transformer_aggregation(
    get_mini_batch: Callable,
    batch_size: int,
    channels: int,
    device: torch.device,
) -> None:
    batch = get_mini_batch(batch_size, channels=channels, device=device)

    aggr = 'patch_transformer'
    aggr_kwargs = dict(
        patch_size=4,
        num_transformer_blocks=2,
        dropout=0.0,
        aggr='mean',
        heads=1,
    )

    model = HeteroGNN(
        metadata=batch.metadata(),
        channels=channels,
        num_mp_layers=2,
        aggr=aggr,
        aggr_kwargs=aggr_kwargs,
        device=device,
    )

    cfg = model.config()
    assert cfg.aggr == aggr
    assert cfg.aggr_kwargs == aggr_kwargs

    out_dict = model(
        batch.x_dict,
        batch.edge_index_dict,
        batch.num_sampled_nodes_dict,
        batch.num_sampled_edges_dict,
    )
    for out in out_dict.values():
        assert out.device == device
        assert out.size(-1) == channels


@withCUDA
def test_hetero_gnn_single_node(device: torch.device) -> None:
    data = HeteroData()
    data['v1'].x = torch.randn(1, 8, device=device)
    edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
    data['v1', 'v1'].edge_index = edge_index

    model = HeteroGNN(
        channels=8,
        num_mp_layers=1,
        norm='batch_norm',
        metadata=data.metadata(),
        device=device,
    )

    # We cannot compute batch statistics for a single node.
    # As such, we expect that `running_mean` and `running_var` are picked up:
    out_dict = model(data.x_dict, data.edge_index_dict)
    assert out_dict['v1'].size() == (1, 8)


@withCUDA
def test_hetero_gnn_no_edges(device: torch.device) -> None:
    data = HeteroData()
    data['v1'].x = torch.randn(1, 8, device=device)

    model = HeteroGNN(
        channels=8,
        num_mp_layers=1,
        metadata=data.metadata(),
        device=device,
    )

    out_dict = model(data.x_dict, {})
    assert out_dict['v1'].size() == (1, 8)


@withCUDA
@pytest.mark.parametrize('batch_size', [64])
@pytest.mark.parametrize('channels', [16])
@pytest.mark.parametrize('num_mp_layers', [0, 1, 2])
def test_trim_to_layer(
    get_mini_batch: Callable,
    batch_size: int,
    channels: int,
    num_mp_layers: int,
    device: torch.device,
) -> None:
    batch = get_mini_batch(batch_size, channels=channels, device=device)

    model = HeteroGNN(
        metadata=batch.metadata(),
        channels=channels,
        num_mp_layers=num_mp_layers,
        device=device,
    )

    x_dict = batch.x_dict
    edge_index_dict = batch.edge_index_dict
    num_sampled_nodes_dict = batch.num_sampled_nodes_dict
    num_sampled_edges_dict = batch.num_sampled_edges_dict

    # simulate sampling up to num_mp_layers hops
    for key in list(num_sampled_edges_dict.keys()):
        num_sampled_edges_dict[key] = num_sampled_edges_dict[
            key][:num_mp_layers]
        if sum(num_sampled_edges_dict[key]) == 0:
            del num_sampled_edges_dict[key]
            del edge_index_dict[key]

    for key in list(num_sampled_nodes_dict.keys()):
        num_sampled_nodes_dict[key] = num_sampled_nodes_dict[
            key][:num_mp_layers + 1]
        if sum(num_sampled_nodes_dict[key]) == 0:
            del num_sampled_nodes_dict[key]
            del x_dict[key]

    out1 = model(batch.x_dict, batch.edge_index_dict)['user'][:batch_size]
    out2 = model(
        x_dict,
        edge_index_dict,
        num_sampled_nodes_dict,
        num_sampled_edges_dict,
    )['user']

    assert torch.allclose(out1, out2, atol=1e-6)


@withCUDA
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('channels', [16])
@pytest.mark.parametrize('num_mp_layers', [2])
def test_trim_to_layer_single_table(
    batch_size: int,
    channels: int,
    num_mp_layers: int,
    device: torch.device,
) -> None:
    batch = HeteroData()
    batch['v1'].x = torch.randn(batch_size, channels, device=device)
    batch['v1'].num_sampled_nodes = [batch_size]

    model = HeteroGNN(
        metadata=batch.metadata(),
        channels=channels,
        num_mp_layers=num_mp_layers,
        device=device,
    )

    out1 = model(batch.x_dict, batch.collect(
        'edge_index',
        allow_empty=True,
    ))['v1'][:batch_size]
    out2 = model(
        batch.x_dict,
        batch.collect('edge_index', allow_empty=True),
        batch.num_sampled_nodes_dict,
        batch.collect('num_sampled_edges', allow_empty=True),
    )['v1']

    assert torch.allclose(out1, out2, atol=1e-6)


@onlyFullTest
@withCUDA
@pytest.mark.skipif(not WITH_PT24, reason='Requires PyTorch 2.4')
def test_compile(device: torch.device, get_mini_batch: Callable) -> None:
    batch = get_mini_batch(64, channels=16, device=device)
    model = HeteroGNN(
        metadata=batch.metadata(),
        channels=16,
        num_mp_layers=2,
        device=device,
    )
    explanation = torch._dynamo.explain(model)(
        batch.x_dict,
        batch.edge_index_dict,
        batch.num_sampled_nodes_dict,
        batch.num_sampled_edges_dict,
    )
    assert explanation.graph_break_count == 0

    compiled_model = torch.compile(model)
    # Note: We call forward/backward twice on purpose. The second call uses
    # 'guards' cached during the first call. The second call is to catch
    # unexpected errors like https://github.com/pytorch/pytorch/issues/111603.
    for _ in range(2):
        out_dict = compiled_model(
            batch.x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        out_dict['trans'].sum().backward(retain_graph=True)
        out_dict['store'].sum().backward(retain_graph=True)
        out_dict['user'].sum().backward()
