# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import cast

import pytest
import torch
from kumoapi.task import TaskType
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.nn.module_dict import ModuleDict

from kumo_core.explain.walk_relevance import (
    HybridLinkReadout,
    NodeReadout,
    WalkRelevance,
    explain_hybrid_link,
)
from kumo_core.modeling.gnn.hetero import HeteroGNN

C = 8
METADATA = (
    ['customers', 'products', 'purchases'],
    [
        ('purchases', 'customer_id', 'customers'),
        ('customers', 'rev_customer_id', 'purchases'),
        ('purchases', 'product_id', 'products'),
        ('products', 'rev_product_id', 'purchases'),
    ],
)
# Purchase i links customer CUSTOMER[i] to product PRODUCT[i]:
CUSTOMER = [0, 0, 0, 1, 1, 2, 2, 0, 1]
PRODUCT = [0, 1, 2, 2, 3, 4, 0, 3, 1]


def make_graph() -> tuple[dict[str, Tensor], dict[tuple, Tensor]]:
    purchase = torch.arange(len(CUSTOMER))
    customer = torch.tensor(CUSTOMER)
    product = torch.tensor(PRODUCT)
    x_dict = {
        'customers': torch.randn(3, C),
        'products': torch.randn(5, C),
        'purchases': torch.randn(len(CUSTOMER), C),
    }
    edge_index_dict = {
        METADATA[1][0]: torch.stack([purchase, customer]),
        METADATA[1][1]: torch.stack([customer, purchase]),
        METADATA[1][2]: torch.stack([purchase, product]),
        METADATA[1][3]: torch.stack([product, purchase]),
    }
    return x_dict, edge_index_dict


def make_gnn(**kwargs) -> HeteroGNN:
    # Mirrors the link prediction configuration used in production:
    defaults = dict(num_pre_mp_layers=1, aggr=['sum', 'mean', 'max'],
                    act='gelu', norm='layer_norm', learn_skip=True)
    defaults.update(kwargs)
    return HeteroGNN(C, 2, metadata=METADATA, **defaults).eval()


def make_mlp(num_layers: int = 2, bias: bool = True) -> MLP:
    return MLP(in_channels=C, hidden_channels=C, out_channels=1,
               num_layers=num_layers, act='gelu', norm='layer_norm',
               norm_kwargs=dict(mode='node'), bias=bias).eval()


def remove_biases(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith('bias'):
                param.zero_()


@pytest.mark.parametrize('method', ['gi', 'lrp'])
def test_rewrite_preserves_forward(method: str) -> None:
    torch.manual_seed(0)
    gnn = make_gnn()
    x_dict, edge_index_dict = make_graph()
    expected = gnn(x_dict, edge_index_dict)
    out = WalkRelevance(gnn, method, gamma=0.25).forward(
        x_dict, edge_index_dict)[-1]
    for key in expected:
        assert torch.allclose(out[key], expected[key], atol=1e-5)


@pytest.mark.parametrize('method', ['gi', 'lrp'])
def test_walks_follow_graph_structure(method: str) -> None:
    torch.manual_seed(0)
    gnn = make_gnn()
    x_dict, edge_index_dict = make_graph()
    readout = NodeReadout(('customers', 0), make_mlp())
    out = WalkRelevance(gnn, method).explain(x_dict, edge_index_dict, readout)

    # Customer 0 has purchases {0, 1, 2, 7}. Walks of three nodes ending at
    # customer 0: 5 middle nodes (itself and its purchases), each with
    # 5 (customer) or 3 (purchase: itself, customer, product) start nodes.
    assert len(out.walks) == 5 + 4 * 3
    assert len(set(out.walks)) == len(out.walks)
    assert all(walk[-1] == ('customers', 0) for walk in out.walks)
    assert (('products', 1), ('purchases', 1),
            ('customers', 0)) in out.walks


@pytest.mark.parametrize('method', ['gi', 'lrp'])
def test_pooled_walks_match_node_relevance(method: str) -> None:
    torch.manual_seed(1)
    gnn = make_gnn()
    x_dict, edge_index_dict = make_graph()
    readout = NodeReadout(('customers', 1), make_mlp())
    explainer = WalkRelevance(gnn, method, gamma=0.25)
    out = explainer.explain(x_dict, edge_index_dict, readout)
    node_rel = explainer.node_relevance(x_dict, edge_index_dict, readout)

    pooled = out.node_relevance(position=0)
    for node_type, rel in node_rel.items():
        for i, value in enumerate(rel.tolist()):
            assert pooled.get((node_type, i), 0.0) == pytest.approx(
                value, abs=1e-6)


@pytest.mark.parametrize('method,act,norm', [
    ('gi', 'relu', None),
    ('lrp', 'relu', None),
    ('lrp', 'gelu', 'layer_norm'),
])
def test_conservation_without_biases(method: str, act: str,
                                     norm: str | None) -> None:
    # Without biases, walk relevance sums to the output: exactly for GI on
    # positively homogeneous models, and for LRP under the gate and frozen
    # layer norm rules.
    torch.manual_seed(2)
    gnn = make_gnn(act=act, norm=norm, bias=False)
    remove_biases(gnn)
    mlp = make_mlp(num_layers=1, bias=False)
    x_dict, edge_index_dict = make_graph()
    readout = NodeReadout(('customers', 0), mlp)
    out = WalkRelevance(gnn, method, gamma=0.5,
                        epsilon=1e-9).explain(x_dict, edge_index_dict,
                                              readout)
    assert out.total == pytest.approx(out.score, rel=1e-4, abs=1e-5)
    assert out.conservation_error < 1e-4


@pytest.mark.parametrize('method', ['gi', 'lrp'])
def test_planted_walk_ranks_top(method: str) -> None:
    # Only channel 0 carries signal, along products -> purchases ->
    # customers. Product 3 has the largest signal and reaches customer 0 via
    # purchase 7, so that walk must be the most relevant one.
    torch.manual_seed(3)
    gnn = make_gnn(num_pre_mp_layers=0, act='relu', norm=None)
    with torch.no_grad():
        for param in gnn.parameters():
            param.mul_(0.01)
        convs = [cast(ModuleDict, convs) for convs in gnn.hetero_convs]
        lin0 = cast(torch.nn.Linear, convs[0][METADATA[1][3]].lin)
        lin0.weight[0, 0] = 1.0
        # Channel 0 of the `sum` block of customer aggregation:
        lin1 = cast(torch.nn.Linear, convs[1][METADATA[1][0]].lin)
        lin1.weight[0, 0] = 1.0

    x_dict, edge_index_dict = make_graph()
    for x in x_dict.values():
        x[:, 0] = 0.0
    x_dict['products'][3, 0] = 3.0
    x_dict['products'][1, 0] = 1.0

    mlp = make_mlp(num_layers=1)
    lin = cast(torch.nn.Linear, mlp.lins[0])
    with torch.no_grad():
        lin.weight.zero_()
        lin.weight[0, 0] = 1.0
        lin.bias.zero_()

    readout = NodeReadout(('customers', 0), mlp)
    out = WalkRelevance(gnn, method).explain(x_dict, edge_index_dict, readout)
    (top_walk, top_rel), (second_walk, _) = out.topk(2)
    assert top_walk == (('products', 3), ('purchases', 7), ('customers', 0))
    assert second_walk == (('products', 1), ('purchases', 1), ('customers',
                                                               0))
    assert top_rel > 0.5 * out.score


class FakeRHSEmbedding(torch.nn.Module):
    lookup_embedding = None  # No RHS-awareness in the GNN input.

    def __init__(self, table: torch.nn.Embedding) -> None:
        super().__init__()
        self.table = table

    def forward(self, index: Tensor) -> Tensor:
        return self.table(index)


class FakeHybridModule(torch.nn.Module):
    def __init__(self, num_rhs: int, dim: int = 4) -> None:
        super().__init__()
        self.distance_measure = 'dot_product'
        self.lhs_projector = torch.nn.Linear(C, dim)
        self.lin_offset_embgnn = torch.nn.Linear(C, 1)
        self.head = torch.nn.Module()
        self.head.mlp_head = make_mlp(num_layers=1)
        self.table = torch.nn.Embedding(num_rhs, dim)
        self.rhs_embedding = FakeRHSEmbedding(self.table)


@pytest.mark.parametrize('method', ['gi', 'lrp'])
def test_link_relevance_is_candidate_specific(method: str) -> None:
    torch.manual_seed(4)
    gnn = make_gnn()
    x_dict, edge_index_dict = make_graph()
    module = FakeHybridModule(num_rhs=2)
    with torch.no_grad():
        module.table.weight[1] = -module.table.weight[0]
        module.lin_offset_embgnn.weight.zero_()
        module.lin_offset_embgnn.bias.zero_()

    explainer = WalkRelevance(gnn, method, gamma=0.25)
    out_a = explainer.explain(
        x_dict, edge_index_dict,
        HybridLinkReadout(module, ('customers', 0), rhs_id=0))
    out_b = explainer.explain(
        x_dict, edge_index_dict,
        HybridLinkReadout(module, ('customers', 0), rhs_id=1))

    assert out_a.walks == out_b.walks
    assert out_a.score == pytest.approx(-out_b.score, abs=1e-6)
    assert torch.allclose(out_a.relevance, -out_b.relevance, atol=1e-6)
    assert (out_a.relevance > 0).any() and (out_a.relevance < 0).any()


def test_link_relevance_id_gnn_branch() -> None:
    torch.manual_seed(5)
    gnn = make_gnn()
    x_dict, edge_index_dict = make_graph()
    module = FakeHybridModule(num_rhs=5)
    h = gnn(x_dict, edge_index_dict)
    h_lhs, h_rhs = h['customers'][0], h['products'][2]
    with torch.no_grad():
        emb_score = (module.lhs_projector(h_lhs) * module.table.weight[2]
                     ).sum() + module.lin_offset_embgnn(h_lhs).sum()
        id_score = module.head.mlp_head(h_rhs.view(1, -1)).sum() + \
            (h_lhs * h_rhs).sum()
        # Force the ID-GNN branch to win:
        module.lin_offset_embgnn.bias -= (emb_score - id_score).abs() + 1.0
    readout = HybridLinkReadout(module, ('customers', 0), rhs_id=2,
                                rhs_node=('products', 2))
    out = WalkRelevance(gnn, 'lrp').explain(x_dict, edge_index_dict, readout)

    assert readout.branch == 'id_gnn'
    assert out.score == pytest.approx(float(id_score), abs=1e-5)
    ends = {walk[-1] for walk in out.walks}
    assert ends == {('customers', 0), ('products', 2)}


def test_prefix_pruning_drops_walks() -> None:
    torch.manual_seed(6)
    gnn = make_gnn()
    x_dict, edge_index_dict = make_graph()
    readout = NodeReadout(('customers', 0), make_mlp())
    full = WalkRelevance(gnn, 'lrp').explain(x_dict, edge_index_dict, readout)
    pruned = WalkRelevance(gnn, 'lrp', min_prefix_relevance=10.0).explain(
        x_dict, edge_index_dict, readout)
    assert len(pruned.walks) < len(full.walks)


def test_unsupported_aggregation() -> None:
    gnn = HeteroGNN(C, 2, metadata=METADATA, aggr=['sum', 'std']).eval()
    with pytest.raises(NotImplementedError, match='std'):
        WalkRelevance(gnn)


class FakeEncoder(torch.nn.Module):
    def forward(self, tf_dict, batch_dict, time_dict, seed_time):
        return dict(tf_dict)


def test_explain_hybrid_link_restricts_to_seed() -> None:
    torch.manual_seed(7)
    gnn = make_gnn()
    x_dict, edge_index_dict = make_graph()

    # Two disjoint copies of the graph; seed 1 is customer 0 of copy 1.
    batch = HeteroData()
    for key, x in x_dict.items():
        n = x.size(0)
        noise = torch.randn_like(x)
        # Seed nodes come first in the LHS table:
        order = [0, 3, 1, 2, 4, 5] if key == 'customers' else None
        x2 = torch.cat([x, noise])
        b = torch.cat([torch.zeros(n), torch.ones(n)]).long()
        n_id = torch.cat([torch.arange(n), torch.arange(n)])
        if order is not None:
            x2, b, n_id = x2[order], b[order], n_id[order]
        batch[key].tf = x2
        batch[key].batch = b
        batch[key].n_id = n_id
    inv = {k: torch.argsort(torch.tensor([0, 3, 1, 2, 4, 5]))
           if k == 'customers' else torch.arange(2 * x_dict[k].size(0))
           for k in x_dict}
    for (src, rel, dst), edge_index in edge_index_dict.items():
        n_src, n_dst = x_dict[src].size(0), x_dict[dst].size(0)
        both = torch.cat([edge_index, edge_index +
                          torch.tensor([[n_src], [n_dst]])], dim=1)
        batch[src, rel, dst].edge_index = torch.stack(
            [inv[src][both[0]], inv[dst][both[1]]])
    batch['customers'].batch_size = 2

    module = FakeHybridModule(num_rhs=5)
    module.model = gnn
    module.encoder = FakeEncoder()
    module.train_table_name = 'train'
    module.task_type = TaskType.STATIC_LINK_PREDICTION
    module.lhs_name, module.rhs_name = 'customers', 'products'
    module.identity_awareness = torch.nn.Embedding(1, C)
    with torch.no_grad():
        module.identity_awareness.weight.zero_()

    out = explain_hybrid_link(module, batch, seed=0, rhs_ids=[2, 4])
    expected = WalkRelevance(gnn, 'lrp').explain(
        x_dict, edge_index_dict,
        HybridLinkReadout(module, ('customers', 0), rhs_id=2,
                          rhs_node=('products', 2)))
    res = out[2]
    assert set(out) == {2, 4}
    assert res.score == pytest.approx(expected.score, abs=1e-5)
    assert torch.allclose(res.relevance, expected.relevance, atol=1e-5)
    # Walks index the batch (customers of copy 0 were reordered):
    assert res.walks == [
        tuple((key, int(inv[key][i])) for key, i in walk)
        for walk in expected.walks
    ]
    assert all(int(batch[key].batch[i]) == 0 for walk in res.walks
               for key, i in walk)
