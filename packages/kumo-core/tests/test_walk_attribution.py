# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from kumo_core.explain.walk_attribution import (
    WalkAttributionConfig, WalkCapture, UnsupportedWalkAttribution,
)
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead


def fixture():
    module = torch.nn.Module()
    module.entity_table_name = 'product'
    module.model = HeteroGNN(1, 1, metadata=(['product'], []), aggr='sum',
                            learn_skip=True, norm=None, bias=True)
    module.head = NodePredHead(1, 1, num_layers=1, norm=None, param_keys=[])
    module.double().eval()
    with torch.no_grad():
        module.model.root_skips[0].lin_dict['product'].weight.fill_(2)
        module.model.root_skips[0].lin_dict['product'].bias.fill_(1)
        module.head.mlp.lins[0].weight.fill_(3)
        module.head.mlp.lins[0].bias.fill_(4)
    return module


def forward(module):
    return module.head(module.model({'product': torch.tensor([[2.]], dtype=torch.float64)}, {})['product'])


def test_capture_bias_ledger_and_no_mutation():
    module = fixture()
    expected = forward(module).detach().clone()
    with WalkCapture(module) as capture:
        actual = forward(module)
        result = capture.result()
    assert torch.equal(actual, expected)
    # y = 3*(2*2 + 1) + 4 = 19, source=12, biases=7.
    assert result['score'] == 19
    assert sum(w['relevance'] for w in result['walks']) == pytest.approx(12)
    assert result['bias_relevance'] == pytest.approx(7)
    assert result['accounting_error'] == pytest.approx(0, abs=1e-12)
    assert not capture.handles
    assert all(not m._forward_hooks for m in module.modules())


def test_budget_reports_omitted_signed_mass():
    module = fixture()
    with WalkCapture(module, WalkAttributionConfig(max_walks=1)) as capture:
        forward(module)
        result = capture.result()
    assert result['truncated'] and not result['complete']
    assert result['omitted_signed_relevance'] == pytest.approx(15)
    assert result['accounting_error'] == pytest.approx(0, abs=1e-12)


def test_unsupported_before_hooks():
    module = fixture()
    module.model.act = 'gelu'
    with pytest.raises(UnsupportedWalkAttribution):
        with WalkCapture(module):
            pass
    assert all(not m._forward_hooks for m in module.modules())


@pytest.mark.parametrize('kwargs', [{'gamma': float('nan')}, {'epsilon': 0},
                                    {'max_walks': True}, {'output_index': -1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        WalkAttributionConfig(**kwargs)


def test_signed_readout_and_bias_cancellation():
    module = fixture()
    with torch.no_grad():
        module.head.mlp.lins[0].weight.fill_(-3)
    with WalkCapture(module) as capture:
        forward(module)
        result = capture.result()
    assert result['score'] == -11
    assert sum(w['relevance'] for w in result['walks']) == pytest.approx(-12)
    assert result['bias_relevance'] == pytest.approx(1)
    assert result['accounting_error'] == pytest.approx(0, abs=1e-12)


def test_capture_failure_preserves_second_prediction():
    module = fixture()
    with WalkCapture(module) as capture:
        first = forward(module)
        second = forward(module)
        assert torch.equal(first, second)
        with pytest.raises(ValueError, match='capture failed'):
            capture.result()
    assert all(not m._forward_hooks for m in module.modules())


def test_inspection_has_no_hooks():
    from kumo_core.explain.walk_attribution import inspect_walk_capability
    module = fixture()
    result = inspect_walk_capability(module)
    assert result['supported'] and result['device_qualified']
    assert all(not m._forward_hooks for m in module.modules())


def test_selected_logit_in_multiseed_forward():
    module = fixture()
    module.head = NodePredHead(1, 2, num_layers=1, norm=None, param_keys=[]).double().eval()
    with torch.no_grad():
        module.head.mlp.lins[0].weight.copy_(torch.tensor([[3.], [-3.]]))
        module.head.mlp.lins[0].bias.fill_(4)
    with WalkCapture(module, WalkAttributionConfig(output_index=1)) as capture:
        module.head(module.model({'product': torch.tensor([[2.], [5.]], dtype=torch.float64)}, {})['product'])
        result = capture.result(seed_index=1)
    assert result['score'] == -29
    assert {w['source_occurrence'] for w in result['walks']} == {1}
    assert sum(w['relevance'] for w in result['walks']) == pytest.approx(-30)
    assert result['bias_relevance'] == pytest.approx(1)


# v1 byte identity (contract sections 4.1 and 12) ##########################
# The generator below uses only APIs that exist at 1909c5b9. The literals in
# `_V1_GOLDEN` were produced by running it under the installed 1909c5b9 wheel
# without the source overlay (ledger/pods/P2-binary/golden/generate.py). A
# failure is never fixed by editing a literal: v1 bytes are frozen.

def _golden_reference(aggregation, depth, sign=1.0):
    from torch import nn
    relation = ('events', 'entity_id', 'entities')
    module = nn.Module()
    module.entity_table_name = 'entities'
    module.model = HeteroGNN(channels=1, num_mp_layers=depth,
                             metadata=(['events', 'entities'], [relation]),
                             aggr=aggregation, act='relu', norm=None,
                             bias=False, learn_skip=False)
    module.head = NodePredHead(in_channels=1, out_channels=1, num_layers=1,
                               norm=None, param_keys=[])
    module.double().eval()
    with torch.no_grad():
        for param in module.model.parameters():
            param.fill_(1)
        module.head.mlp.lins[0].weight.fill_(3 * sign)
        module.head.mlp.lins[0].bias.zero_()
    x = {'events': torch.tensor([[2.], [4.]], dtype=torch.float64),
         'entities': torch.tensor([[1.]], dtype=torch.float64)}
    edges = {relation: torch.tensor([[0, 1], [0, 0]])}
    sample = {'nodes': {'events': {'pkeys': [11, 12]},
                        'entities': {'pkeys': [7]}}}
    return module, x, edges, sample


def _golden_b1():
    # P0 fixture B1: two layers with Engine trimming; f = 9.
    fk = ('transactions', 'customer_id', 'customer')
    rev = ('customer', 'rev_customer_id', 'transactions')
    art_rev = ('article', 'rev_article_id', 'transactions')
    art_fk = ('transactions', 'article_id', 'article')
    module = torch.nn.Module()
    module.entity_table_name = 'customer'
    module.model = HeteroGNN(1, 2, metadata=(['customer', 'transactions',
                                              'article'],
                                             [fk, rev, art_rev, art_fk]),
                             aggr=['mean'], norm=None, bias=False,
                             learn_skip=True)
    module.head = NodePredHead(1, 1, num_layers=1, norm=None, param_keys=[])
    module.head.mlp.lins[0].register_parameter('bias', None)
    module.double().eval()
    with torch.no_grad():
        for param in module.parameters():
            param.fill_(1.0)
    x = {'customer': torch.tensor([[1.]], dtype=torch.float64),
         'transactions': torch.tensor([[2.], [3.]], dtype=torch.float64),
         'article': torch.tensor([[4.]], dtype=torch.float64)}
    edges = {fk: torch.tensor([[0, 1], [0, 0]]),
             rev: torch.tensor([[0, 0], [0, 1]]),
             art_rev: torch.tensor([[0], [0]]),
             art_fk: torch.zeros(2, 0, dtype=torch.long)}
    nodes = {'customer': [1, 0, 0], 'transactions': [0, 2, 0],
             'article': [0, 0, 1]}
    nse = {fk: [2, 0], rev: [0, 2], art_rev: [0, 1], art_fk: [0, 0]}

    def run():
        out = module.model(dict(x), dict(edges),
                           {k: list(v) for k, v in nodes.items()},
                           {k: list(v) for k, v in nse.items()})
        return module.head(out['customer'][:1])

    sample = {'nodes': {'customer': {'pkeys': [17]},
                        'transactions': {'pkeys': [41, 42]},
                        'article': {'pkeys': [9]}}}
    return module, run, sample


def _golden_preprocessing():
    # One pre-MP layer, learned roots, two head layers, signed weights and
    # biases, two channels, dyadic float64 values.
    relation = ('events', 'entity_id', 'entities')
    module = torch.nn.Module()
    module.entity_table_name = 'entities'
    module.model = HeteroGNN(channels=2, num_mp_layers=1,
                             metadata=(['events', 'entities'], [relation]),
                             num_pre_mp_layers=1, aggr='mean', act='relu',
                             norm=None, bias=True, learn_skip=True)
    module.head = NodePredHead(in_channels=2, out_channels=1,
                               hidden_channels=2, num_layers=2, norm=None,
                               param_keys=[])
    module.double().eval()

    def put(param, values):
        param.copy_(torch.tensor(values, dtype=torch.float64))

    with torch.no_grad():
        pre = module.model.lins[0].lin_dict
        put(pre['events'].weight, [[1., -0.5], [0.25, 1.]])
        put(pre['events'].bias, [0.5, -0.25])
        put(pre['entities'].weight, [[0.5, 0.5], [-1., 1.]])
        put(pre['entities'].bias, [0.125, 0.25])
        conv = module.model.hetero_convs[0][relation].lin
        put(conv.weight, [[1., 0.5], [-0.5, 1.]])
        put(conv.bias, [0.25, -0.5])
        root = module.model.root_skips[0].lin_dict
        put(root['events'].weight, [[0.5, 0.], [0., 0.5]])
        put(root['events'].bias, [0., 0.125])
        put(root['entities'].weight, [[1., -0.25], [0.5, 0.75]])
        put(root['entities'].bias, [-0.125, 0.5])
        put(module.head.mlp.lins[0].weight, [[1., -1.], [0.5, 0.5]])
        put(module.head.mlp.lins[0].bias, [0.5, -0.5])
        put(module.head.mlp.lins[1].weight, [[2., -1.]])
        put(module.head.mlp.lins[1].bias, [0.25])
    x = {'events': torch.tensor([[1., 2.], [0.5, 1.5], [2., 0.]],
                                dtype=torch.float64),
         'entities': torch.tensor([[1., 0.5]], dtype=torch.float64)}
    edges = {relation: torch.tensor([[0, 1, 2], [0, 0, 0]])}
    sample = {'nodes': {'events': {'pkeys': [21, 22, 23]},
                        'entities': {'pkeys': [5]}}}
    return module, x, edges, sample


def v1_golden_artifacts():
    """Return {name: exact JSON text} for frozen v1 capture outputs."""
    from kumo_core.explain.local_sensitivity import _encoded, bind_walk_result
    from kumo_core.explain.walk_attribution import inspect_walk_capability
    out = {}

    def capture(name, module, run, sample, seed_index=0, **config):
        with WalkCapture(module, WalkAttributionConfig(**config)) as cap:
            run()
            result = cap.result(seed_index=seed_index)
        out[name] = _encoded(bind_walk_result(result, sample))

    module = fixture()
    product = {'nodes': {'product': {'pkeys': ['p17']}}}
    capture('capture_19', module, lambda: forward(module), product)
    capture('capture_19_max_walks_1', module, lambda: forward(module),
            product, max_walks=1)
    signed = fixture()
    with torch.no_grad():
        signed.head.mlp.lins[0].weight.fill_(-3)
    capture('capture_19_signed_readout', signed, lambda: forward(signed),
            product)
    for aggregation in ('sum', 'mean'):
        for depth in (1, 2):
            for sign in (1.0, -1.0):
                m, x, e, s = _golden_reference(aggregation, depth, sign)
                capture(f'reference_{aggregation}_d{depth}_s{int(sign)}', m,
                        lambda: m.head(m.model(x, e)['entities'], None), s,
                        epsilon=1e-12)
    m, x, e, s = _golden_reference('sum', 2)
    capture('reference_sum_d2_top_k_1', m,
            lambda: m.head(m.model(x, e)['entities'], None), s, top_k=1)
    m, x, e, s = _golden_reference('sum', 1)
    x['events'][0] = -2
    capture('reference_gamma_1_signed_input', m,
            lambda: m.head(m.model(x, e)['entities'], None), s, gamma=1.,
            epsilon=1e-12)
    m, run, s = _golden_b1()
    capture('b1_default', m, run, s)
    capture('b1_gamma_025', m, run, s, gamma=0.25)
    capture('b1_max_walks_3', m, run, s, max_walks=3)
    capture('b1_max_walks_8', m, run, s, max_walks=8)
    capture('b1_top_k_3', m, run, s, top_k=3)
    m, x, e, s = _golden_preprocessing()
    capture('preprocessing_head2', m,
            lambda: m.head(m.model(x, e)['entities'], None), s)
    capture('preprocessing_head2_gamma_025', m,
            lambda: m.head(m.model(x, e)['entities'], None), s, gamma=0.25)
    multi = fixture()
    multi.head = NodePredHead(1, 2, num_layers=1, norm=None,
                              param_keys=[]).double().eval()
    with torch.no_grad():
        multi.head.mlp.lins[0].weight.copy_(torch.tensor([[3.], [-3.]]))
        multi.head.mlp.lins[0].bias.fill_(4)
    capture('multiseed_output_index_1', multi,
            lambda: multi.head(multi.model({'product': torch.tensor(
                [[2.], [5.]], dtype=torch.float64)}, {})['product']),
            {'nodes': {'product': {'pkeys': ['p17', 'p18']}}},
            seed_index=1, output_index=1)
    out['inspect_supported'] = _encoded(inspect_walk_capability(fixture()))
    unsupported = fixture()
    unsupported.model.act = 'gelu'
    out['inspect_unsupported'] = _encoded(
        inspect_walk_capability(unsupported))
    out['default_modules_seed_11'] = _golden_default_modules()
    return out


def _golden_default_modules():
    # The default modules AutoTrainer builds, keys, shapes and same-seed
    # initial values (contract 8: identical to 1909c5b9 by default).
    import hashlib
    from types import SimpleNamespace

    from kumo_core.automl.auto_trainer import AutoTrainer
    from kumo_core.explain.local_sensitivity import _encoded
    from kumoapi.task import TaskType
    stub = AutoTrainer.__new__(AutoTrainer)
    stub.task = SimpleNamespace(task_type=TaskType.BINARY_CLASSIFICATION)
    cfg = SimpleNamespace(
        model_architecture=SimpleNamespace(
            model=[SimpleNamespace(channels=[8], aggregation=[['mean']],
                                   dropout=[0.0])],
            activation=['relu'], normalization=[None],
            num_pre_message_passing_layers=[1],
            num_post_message_passing_layers=[2],
            prediction_time_encodings=[False]),
        neighbor_sampling=SimpleNamespace(num_neighbors=[[2, 2]]))
    torch.manual_seed(11)
    gnn = stub._get_model(cfg)
    gnn.metadata = (['events', 'entities'],
                    [('events', 'entity_id', 'entities'),
                     ('entities', 'rev_entity_id', 'events')])
    head = stub._get_head(cfg)
    head.in_channels, head.out_channels, head.param_keys = 8, 1, []
    state = {**{f'model.{k}': v for k, v in gnn.state_dict().items()},
             **{f'head.{k}': v for k, v in head.state_dict().items()}}
    return _encoded({key: [list(value.shape), hashlib.sha256(
        value.detach().contiguous().numpy().tobytes()).hexdigest()]
                     for key, value in state.items()})


_V1_GOLDEN = {
    'b1_default':
    '{"accounting_error":0.0,"bias_relevance":0.0,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":12,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":9.0,"search":"bounded_exhaustive","stabilizer_residual":4.000000330961484e-09,"status":"available","truncated":false,"walks":[{"relevance":1.9999999992698412,"source_occurrence":0,"source_pkey":9,"source_table":"article","steps":[{"destination_occurrence":0,"destination_pkey":41,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":9,"type":["article","rev_article_id","transactions"]},{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":41,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":1.4999999992916666,"source_occurrence":1,"source_pkey":42,"source_table":"transactions","steps":[{"kind":"self","layer":0,"occurrence":1,"pkey":42,"table":"transactions"},{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":1,"kind":"relation","layer":1,"source_occurrence":1,"source_pkey":42,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":1.4999999992380952,"source_occurrence":1,"source_pkey":42,"source_table":"transactions","steps":[{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":42,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":1,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":0.9999999996349206,"source_occurrence":0,"source_pkey":41,"source_table":"transactions","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":41,"table":"transactions"},{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":41,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":0.9999999994920634,"source_occurrence":0,"source_pkey":17,"source_table":"customer","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":1,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":0.9999999994920634,"source_occurrence":0,"source_pkey":41,"source_table":"transactions","steps":[{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":41,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":1,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":0.4999999998174603,"source_occurrence":0,"source_pkey":17,"source_table":"customer","steps":[{"destination_occurrence":0,"destination_pkey":41,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":17,"type":["customer","rev_customer_id","transactions"]},{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":41,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":0.49999999976388887,"source_occurrence":0,"source_pkey":17,"source_table":"customer","steps":[{"destination_occurrence":1,"destination_pkey":42,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":17,"type":["customer","rev_customer_id","transactions"]},{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":1,"kind":"relation","layer":1,"source_occurrence":1,"source_pkey":42,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]}]}',
    'b1_gamma_025':
    '{"accounting_error":0.0,"bias_relevance":0.0,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":12,"gamma":0.25,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":9.0,"search":"bounded_exhaustive","stabilizer_residual":3.200000264769187e-09,"status":"available","truncated":false,"walks":[{"relevance":1.999999999415873,"source_occurrence":0,"source_pkey":9,"source_table":"article","steps":[{"destination_occurrence":0,"destination_pkey":41,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":9,"type":["article","rev_article_id","transactions"]},{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":41,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":1.4999999994333333,"source_occurrence":1,"source_pkey":42,"source_table":"transactions","steps":[{"kind":"self","layer":0,"occurrence":1,"pkey":42,"table":"transactions"},{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":1,"kind":"relation","layer":1,"source_occurrence":1,"source_pkey":42,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":1.4999999993904762,"source_occurrence":1,"source_pkey":42,"source_table":"transactions","steps":[{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":42,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":1,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":0.9999999997079365,"source_occurrence":0,"source_pkey":41,"source_table":"transactions","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":41,"table":"transactions"},{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":41,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":0.9999999995936508,"source_occurrence":0,"source_pkey":17,"source_table":"customer","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":1,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":0.9999999995936508,"source_occurrence":0,"source_pkey":41,"source_table":"transactions","steps":[{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":41,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":1,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":0.49999999985396826,"source_occurrence":0,"source_pkey":17,"source_table":"customer","steps":[{"destination_occurrence":0,"destination_pkey":41,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":17,"type":["customer","rev_customer_id","transactions"]},{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":41,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":0.4999999998111111,"source_occurrence":0,"source_pkey":17,"source_table":"customer","steps":[{"destination_occurrence":1,"destination_pkey":42,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":17,"type":["customer","rev_customer_id","transactions"]},{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":1,"kind":"relation","layer":1,"source_occurrence":1,"source_pkey":42,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]}]}',
    'b1_max_walks_3':
    '{"accounting_error":0.0,"bias_relevance":0.0,"complete":false,"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":1,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":8.999999999,"output_index":0,"schema":"walk_attribution/v1","score":9.0,"search":"bounded_exhaustive","stabilizer_residual":1.000000082740371e-09,"status":"truncated","truncated":true,"walks":[]}',
    'b1_max_walks_8':
    '{"accounting_error":0.0,"bias_relevance":0.0,"complete":false,"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":7,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":5.499999998777778,"output_index":0,"schema":"walk_attribution/v1","score":9.0,"search":"bounded_exhaustive","stabilizer_residual":3.000000248221113e-09,"status":"truncated","truncated":true,"walks":[{"relevance":1.4999999992380952,"source_occurrence":1,"source_pkey":42,"source_table":"transactions","steps":[{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":42,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":1,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":0.9999999994920634,"source_occurrence":0,"source_pkey":17,"source_table":"customer","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":1,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":0.9999999994920634,"source_occurrence":0,"source_pkey":41,"source_table":"transactions","steps":[{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":41,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":1,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]}]}',
    'b1_top_k_3':
    '{"accounting_error":4.440892098500626e-16,"bias_relevance":0.0,"complete":false,"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":12,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":3.9999999982003964,"output_index":0,"schema":"walk_attribution/v1","score":9.0,"search":"bounded_exhaustive","stabilizer_residual":4.000000330961484e-09,"status":"truncated","truncated":true,"walks":[{"relevance":1.9999999992698412,"source_occurrence":0,"source_pkey":9,"source_table":"article","steps":[{"destination_occurrence":0,"destination_pkey":41,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":9,"type":["article","rev_article_id","transactions"]},{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":41,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":1.4999999992916666,"source_occurrence":1,"source_pkey":42,"source_table":"transactions","steps":[{"kind":"self","layer":0,"occurrence":1,"pkey":42,"table":"transactions"},{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":1,"kind":"relation","layer":1,"source_occurrence":1,"source_pkey":42,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]},{"relevance":1.4999999992380952,"source_occurrence":1,"source_pkey":42,"source_table":"transactions","steps":[{"destination_occurrence":0,"destination_pkey":17,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":42,"type":["transactions","customer_id","customer"]},{"kind":"self","layer":1,"occurrence":0,"pkey":17,"table":"customer"},{"kind":"self","layer":2,"occurrence":0,"pkey":17,"table":"customer"}]}]}',
    'capture_19':
    '{"accounting_error":-8.881784197001252e-16,"bias_relevance":6.999999999031579,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":2,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":19.0,"search":"bounded_exhaustive","stabilizer_residual":4.000000330961484e-09,"status":"available","truncated":false,"walks":[{"relevance":11.999999996968421,"source_occurrence":0,"source_pkey":"p17","source_table":"product","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":"p17","table":"product"},{"kind":"self","layer":1,"occurrence":0,"pkey":"p17","table":"product"}]}]}',
    'capture_19_max_walks_1':
    '{"accounting_error":0.0,"bias_relevance":3.9999999997894737,"complete":false,"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":1,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":14.999999999210527,"output_index":0,"schema":"walk_attribution/v1","score":19.0,"search":"bounded_exhaustive","stabilizer_residual":1.000000082740371e-09,"status":"truncated","truncated":true,"walks":[]}',
    'capture_19_signed_readout':
    '{"accounting_error":4.440892098500626e-16,"bias_relevance":1.000000000509091,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":2,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":-11.0,"search":"bounded_exhaustive","stabilizer_residual":-4.000000330961484e-09,"status":"available","truncated":false,"walks":[{"relevance":-11.999999996509091,"source_occurrence":0,"source_pkey":"p17","source_table":"product","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":"p17","table":"product"},{"kind":"self","layer":1,"occurrence":0,"pkey":"p17","table":"product"}]}]}',
    'default_modules_seed_11':
    '{"head.mlp.lins.0.bias":[[8],"64e2821e578e8e41be94ea21411b19ea635fcd2080e5dc24c50b3b386376461c"],"head.mlp.lins.0.weight":[[8,8],"da7041772cc5f90ef21c0d765ae03f4695f270eb73b71d8397dc68f224a42d75"],"head.mlp.lins.1.bias":[[1],"9ea8c19b1a544bfa29e95837e053a5c745f502c0bf2d1f645df187d5aa2a6a04"],"head.mlp.lins.1.weight":[[1,8],"c11ac5d6aae2955310df3fd88fdf097f0536560d7f18167d7a951404c34a2e23"],"model.hetero_convs.0.<entities___rev_entity_id___events>.lin.bias":[[8],"89a66a492cbce41c5aa2f7e39f520c2244a3fdfbd233f25d2c97875480b8a515"],"model.hetero_convs.0.<entities___rev_entity_id___events>.lin.weight":[[8,8],"b5adff9a238636e96c448e1915dbe02f92e15e6c2bf7273f7a2a1768b5d60de5"],"model.hetero_convs.0.<events___entity_id___entities>.lin.bias":[[8],"ce5c4f4b46129c3980a95ffa85e7476e29242d01557b6d4d702daefa6bbc66d1"],"model.hetero_convs.0.<events___entity_id___entities>.lin.weight":[[8,8],"fed43419e37fd8e4128070b22449669ca2d20bb36e54c11901e3254ab0896b85"],"model.hetero_convs.1.<entities___rev_entity_id___events>.lin.bias":[[8],"bbd00cb8e1e067135fafaf4158d7e29b4c77aadc1072650881b00de9115c9ab2"],"model.hetero_convs.1.<entities___rev_entity_id___events>.lin.weight":[[8,8],"55f063e79b78135c12559e67a8a5173a908823e8d6e4ce1bfee2cc3e44491acf"],"model.hetero_convs.1.<events___entity_id___entities>.lin.bias":[[8],"941e2494b25e9a667235c0a6a73bdad78e4bb79a810a6def4e076c477b95c1a4"],"model.hetero_convs.1.<events___entity_id___entities>.lin.weight":[[8,8],"b8454289ed8dbbdd234c47b7e882677382cc1dd3cf2ffbaf7938d1bb4d48c13d"],"model.lins.0.lin_dict.entities.bias":[[8],"975eab0c37abe7620451c754bf2d21959286b42dfef68ce55927db3879fa8081"],"model.lins.0.lin_dict.entities.weight":[[8,8],"b38690762774582ed47d1ea3d8be626e17a4be7a7e1e726888023cda7b7f135e"],"model.lins.0.lin_dict.events.bias":[[8],"154065ad7727191a835a13c56fea00b35ba8f99f46010c247a3f3dbf8a74c7e0"],"model.lins.0.lin_dict.events.weight":[[8,8],"2035f91f28912e3aab38668602183a09fd01205d687a1fc17c4c7f5f1a0d9db7"],"model.root_skips.0.lin_dict.entities.bias":[[8],"e74f229647ba6e64d4bd17dc9e867a3cd7af891549e58696be087c1726ef55eb"],"model.root_skips.0.lin_dict.entities.weight":[[8,8],"735d4c4198dfbcfabe88889c9598089e50694187421d704dde93a5c95aae3e78"],"model.root_skips.0.lin_dict.events.bias":[[8],"17a89fe7f1a4f6b0397133a126471986c07ea2da0c76902820ecbd380a4d0559"],"model.root_skips.0.lin_dict.events.weight":[[8,8],"61afb0cd2da4f7d43e74493a41ad590f8fe8c5d7bbf20d4770519328af82c5b8"],"model.root_skips.1.lin_dict.entities.bias":[[8],"f5bac4277c3a4b313d093cf1fbb793aaedea7d9179fedd331b19b23727faaa36"],"model.root_skips.1.lin_dict.entities.weight":[[8,8],"65e62cb18760d7e0002d9db26bbaa7a5046607d765049339124136d950d78b76"],"model.root_skips.1.lin_dict.events.bias":[[8],"bd3a9461e477d06cddbfef8ec13ac5a27b2362e5c806151f62bb433b77af9bfd"],"model.root_skips.1.lin_dict.events.weight":[[8,8],"f5caa9906c538b7acb03064cbfc4e7dfb99955246a2d7a0d2c128836acee32a2"]}',
    'inspect_supported':
    '{"coordinate":"encoded_model_output","device_qualified":true,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","qualification":"reference_operator_profile","qualified_devices":["cpu"],"supported":true}',
    'inspect_unsupported':
    '{"method":"hetero_lrp_gamma_reference_v1","reason":"Reference profile requires ReLU/identity and no normalization","supported":false}',
    'multiseed_output_index_1':
    '{"accounting_error":3.552713678800501e-15,"bias_relevance":1.000000000238245,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":2,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":1,"schema":"walk_attribution/v1","score":-29.0,"search":"bounded_exhaustive","stabilizer_residual":-4.000007436388842e-09,"status":"available","truncated":false,"walks":[{"relevance":-29.99999999623824,"source_occurrence":1,"source_pkey":"p18","source_table":"product","steps":[{"kind":"self","layer":0,"occurrence":1,"pkey":"p18","table":"product"},{"kind":"self","layer":1,"occurrence":1,"pkey":"p18","table":"product"}]}]}',
    'preprocessing_head2':
    '{"accounting_error":1.3877787807814457e-16,"bias_relevance":3.781249991886469,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":10,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":3.020833333333334,"search":"bounded_exhaustive","stabilizer_residual":2.2500009355663764e-09,"status":"available","truncated":false,"walks":[{"relevance":1.541666665122647,"source_occurrence":2,"source_pkey":23,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":2,"pkey":23,"table":"events"},{"destination_occurrence":0,"destination_pkey":5,"edge_occurrence":2,"kind":"relation","layer":1,"source_occurrence":2,"source_pkey":23,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":5,"table":"entities"},{"kind":"self","layer":3,"occurrence":0,"pkey":5,"table":"entities"}]},{"relevance":-1.312499996657471,"source_occurrence":0,"source_pkey":21,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":21,"table":"events"},{"destination_occurrence":0,"destination_pkey":5,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":21,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":5,"table":"entities"},{"kind":"self","layer":3,"occurrence":0,"pkey":5,"table":"entities"}]},{"relevance":-1.1770833294653942,"source_occurrence":1,"source_pkey":22,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":1,"pkey":22,"table":"events"},{"destination_occurrence":0,"destination_pkey":5,"edge_occurrence":1,"kind":"relation","layer":1,"source_occurrence":1,"source_pkey":22,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":5,"table":"entities"},{"kind":"self","layer":3,"occurrence":0,"pkey":5,"table":"entities"}]},{"relevance":0.18750000019708218,"source_occurrence":0,"source_pkey":5,"source_table":"entities","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":5,"table":"entities"},{"kind":"self","layer":1,"occurrence":0,"pkey":5,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":5,"table":"entities"},{"kind":"self","layer":3,"occurrence":0,"pkey":5,"table":"entities"}]}]}',
    'preprocessing_head2_gamma_025':
    '{"accounting_error":9.020562075079397e-16,"bias_relevance":2.3248985988182813,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":10,"gamma":0.25,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":3.020833333333334,"search":"bounded_exhaustive","stabilizer_residual":2.516875582947442e-09,"status":"available","truncated":false,"walks":[{"relevance":1.1371695110119726,"source_occurrence":2,"source_pkey":23,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":2,"pkey":23,"table":"events"},{"destination_occurrence":0,"destination_pkey":5,"edge_occurrence":2,"kind":"relation","layer":1,"source_occurrence":2,"source_pkey":23,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":5,"table":"entities"},{"kind":"self","layer":3,"occurrence":0,"pkey":5,"table":"entities"}]},{"relevance":-0.4606499319940675,"source_occurrence":0,"source_pkey":21,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":21,"table":"events"},{"destination_occurrence":0,"destination_pkey":5,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":21,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":5,"table":"entities"},{"kind":"self","layer":3,"occurrence":0,"pkey":5,"table":"entities"}]},{"relevance":0.45059410353919205,"source_occurrence":0,"source_pkey":5,"source_table":"entities","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":5,"table":"entities"},{"kind":"self","layer":1,"occurrence":0,"pkey":5,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":5,"table":"entities"},{"kind":"self","layer":3,"occurrence":0,"pkey":5,"table":"entities"}]},{"relevance":-0.43117895055892097,"source_occurrence":1,"source_pkey":22,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":1,"pkey":22,"table":"events"},{"destination_occurrence":0,"destination_pkey":5,"edge_occurrence":1,"kind":"relation","layer":1,"source_occurrence":1,"source_pkey":22,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":5,"table":"entities"},{"kind":"self","layer":3,"occurrence":0,"pkey":5,"table":"entities"}]}]}',
    'reference_gamma_1_signed_input':
    '{"accounting_error":0.0,"bias_relevance":0.0,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-12,"expanded_routes":4,"gamma":1.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":9.0,"search":"bounded_exhaustive","stabilizer_residual":1.623590151211829e-12,"status":"available","truncated":false,"walks":[{"relevance":8.999999999998376,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":2.249999999999594,"source_occurrence":0,"source_pkey":7,"source_table":"entities","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-2.249999999999594,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]}]}',
    'reference_mean_d1_s-1':
    '{"accounting_error":0.0,"bias_relevance":0.0,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-12,"expanded_routes":4,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":-12.0,"search":"bounded_exhaustive","stabilizer_residual":-4.000355602329364e-12,"status":"available","truncated":false,"walks":[{"relevance":-5.999999999998,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-2.999999999999,"source_occurrence":0,"source_pkey":7,"source_table":"entities","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-2.999999999999,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]}]}',
    'reference_mean_d1_s1':
    '{"accounting_error":0.0,"bias_relevance":0.0,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-12,"expanded_routes":4,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":12.0,"search":"bounded_exhaustive","stabilizer_residual":4.000355602329364e-12,"status":"available","truncated":false,"walks":[{"relevance":5.999999999998,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":2.999999999999,"source_occurrence":0,"source_pkey":7,"source_table":"entities","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":2.999999999999,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]}]}',
    'reference_mean_d2_s-1':
    '{"accounting_error":3.552713678800501e-15,"bias_relevance":0.0,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-12,"expanded_routes":9,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":-21.0,"search":"bounded_exhaustive","stabilizer_residual":-1.000088900582341e-11,"status":"available","truncated":false,"walks":[{"relevance":-5.999999999997358,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-5.999999999997358,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":1,"pkey":12,"table":"events"},{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":1,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-2.999999999998679,"source_occurrence":0,"source_pkey":7,"source_table":"entities","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-2.999999999998679,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-2.9999999999979288,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":11,"table":"events"},{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]}]}',
    'reference_mean_d2_s1':
    '{"accounting_error":-3.552713678800501e-15,"bias_relevance":0.0,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-12,"expanded_routes":9,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":21.0,"search":"bounded_exhaustive","stabilizer_residual":1.000088900582341e-11,"status":"available","truncated":false,"walks":[{"relevance":5.999999999997358,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":5.999999999997358,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":1,"pkey":12,"table":"events"},{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":1,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":2.999999999998679,"source_occurrence":0,"source_pkey":7,"source_table":"entities","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":2.999999999998679,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":2.9999999999979288,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":11,"table":"events"},{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]}]}',
    'reference_sum_d1_s-1':
    '{"accounting_error":0.0,"bias_relevance":0.0,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-12,"expanded_routes":4,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":-21.0,"search":"bounded_exhaustive","stabilizer_residual":-3.9968028886505635e-12,"status":"available","truncated":false,"walks":[{"relevance":-11.999999999997716,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-5.999999999998858,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-2.999999999999429,"source_occurrence":0,"source_pkey":7,"source_table":"entities","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]}]}',
    'reference_sum_d1_s1':
    '{"accounting_error":0.0,"bias_relevance":0.0,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-12,"expanded_routes":4,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":21.0,"search":"bounded_exhaustive","stabilizer_residual":3.9968028886505635e-12,"status":"available","truncated":false,"walks":[{"relevance":11.999999999997716,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":5.999999999998858,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":2.999999999999429,"source_occurrence":0,"source_pkey":7,"source_table":"entities","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"}]}]}',
    'reference_sum_d2_s-1':
    '{"accounting_error":0.0,"bias_relevance":0.0,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-12,"expanded_routes":9,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":-39.0,"search":"bounded_exhaustive","stabilizer_residual":-1.3002932064409833e-11,"status":"available","truncated":false,"walks":[{"relevance":-11.999999999997055,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-11.999999999995769,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":1,"pkey":12,"table":"events"},{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":1,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-5.999999999998527,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-5.999999999996384,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":11,"table":"events"},{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":-2.9999999999992637,"source_occurrence":0,"source_pkey":7,"source_table":"entities","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]}]}',
    'reference_sum_d2_s1':
    '{"accounting_error":0.0,"bias_relevance":0.0,"complete":true,"coordinate":"encoded_model_output","epsilon":1e-12,"expanded_routes":9,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":0.0,"output_index":0,"schema":"walk_attribution/v1","score":39.0,"search":"bounded_exhaustive","stabilizer_residual":1.3002932064409833e-11,"status":"available","truncated":false,"walks":[{"relevance":11.999999999997055,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":11.999999999995769,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":1,"pkey":12,"table":"events"},{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":1,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":5.999999999998527,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":0,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":5.999999999996384,"source_occurrence":0,"source_pkey":11,"source_table":"events","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":11,"table":"events"},{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":0,"kind":"relation","layer":1,"source_occurrence":0,"source_pkey":11,"type":["events","entity_id","entities"]},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]},{"relevance":2.9999999999992637,"source_occurrence":0,"source_pkey":7,"source_table":"entities","steps":[{"kind":"self","layer":0,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]}]}',
    'reference_sum_d2_top_k_1':
    '{"accounting_error":3.552713678800501e-15,"bias_relevance":0.0,"complete":false,"coordinate":"encoded_model_output","epsilon":1e-09,"expanded_routes":9,"gamma":0.0,"input_coordinate":"encoded_node_features","method":"hetero_lrp_gamma_reference_v1","omitted_signed_relevance":26.999999989945056,"output_index":0,"schema":"walk_attribution/v1","score":39.0,"search":"bounded_exhaustive","stabilizer_residual":1.2999993970197465e-08,"status":"truncated","truncated":true,"walks":[{"relevance":11.999999997054946,"source_occurrence":1,"source_pkey":12,"source_table":"events","steps":[{"destination_occurrence":0,"destination_pkey":7,"edge_occurrence":1,"kind":"relation","layer":0,"source_occurrence":1,"source_pkey":12,"type":["events","entity_id","entities"]},{"kind":"self","layer":1,"occurrence":0,"pkey":7,"table":"entities"},{"kind":"self","layer":2,"occurrence":0,"pkey":7,"table":"entities"}]}]}',
}


def test_v1_bytes_identical_to_1909c5b9():
    regenerated = v1_golden_artifacts()
    assert sorted(regenerated) == sorted(_V1_GOLDEN)
    for name, text in _V1_GOLDEN.items():
        assert regenerated[name] == text, name


# Configuration, registry and schema selection (contract section 3) ##########

LEGACY_KEYS = ['gamma', 'epsilon', 'max_walks', 'top_k', 'output_index']


def test_to_dict_keeps_legacy_bytes_and_round_trips():
    from kumo_core.explain.walk_attribution import DEGREE_T_METHOD
    default = WalkAttributionConfig()
    assert list(default.to_dict()) == LEGACY_KEYS
    assert WalkAttributionConfig(**default.to_dict()) == default
    integer_gamma = WalkAttributionConfig(gamma=1)
    assert type(integer_gamma.to_dict()['gamma']) is int  # as vars() kept it
    extended = WalkAttributionConfig(method=DEGREE_T_METHOD, feature_vectors=True,
                                     bags=True, selected_class=False)
    assert extended.to_dict() == {**dict(zip(LEGACY_KEYS, (0.0, 1e-9, 10000, 20, 0))),
                                  'method': DEGREE_T_METHOD, 'feature_vectors': True,
                                  'bags': True, 'selected_class': False}
    assert WalkAttributionConfig(**extended.to_dict()) == extended
    # vars() now includes the added fields; every serializer uses to_dict().
    assert set(vars(default)) - set(default.to_dict()) == {
        'method', 'feature_vectors', 'bags', 'selected_class'}
    assert WalkAttributionConfig(selected_class=0).to_dict()['selected_class'] == 0


@pytest.mark.parametrize('kwargs,match', [
    ({'method': 'hetero_lrp_mystery_v1'}, 'Unknown walk attribution method'),
    ({'method': None}, 'Unknown walk attribution method'),
    ({'feature_vectors': 1}, 'feature_vectors must be a bool'),
    ({'bags': 'yes'}, 'bags must be a bool'),
    ({'selected_class': 1.5}, 'selected_class'),
    ({'selected_class': [True]}, 'selected_class'),
    ({'method': 'hetero_gnn_gi_degree_t_v1', 'gamma': 0.25}, 'no gamma rule'),
    ({'method': 'hetero_gnn_gi_degree_t_v1', 'epsilon': 1e-6}, 'no gamma rule'),
    ({'bags': True}, 'bags=True requires'),
    ({'method': 'hetero_gnn_lrp_weight_gamma_v1', 'bags': True}, 'bags=True requires'),
])
def test_invalid_extended_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WalkAttributionConfig(**kwargs)


def test_schema_selection_matrix():
    from kumoapi.task import TaskType

    from kumo_core.explain.walk_attribution import select_schema
    regression, binary = TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION
    v1, v2 = 'walk_attribution/v1', 'walk_attribution/v2'
    cases = [
        (WalkAttributionConfig(), regression, v1),
        (WalkAttributionConfig(gamma=0.25, top_k=3, max_walks=7, output_index=1),
         regression, v1),
        (WalkAttributionConfig(), binary, v2),
        (WalkAttributionConfig(feature_vectors=True), regression, v2),
        (WalkAttributionConfig(selected_class=True), regression, v2),
        (WalkAttributionConfig(method='hetero_gnn_lrp_weight_gamma_v1'), regression, v2),
        (WalkAttributionConfig(method='hetero_gnn_gi_degree_t_v1'), regression, v2),
        (WalkAttributionConfig(method='hetero_gnn_gi_degree_t_v1', bags=True), binary, v2),
    ]
    for config, task, schema in cases:
        assert select_schema(config, task) == schema
        assert select_schema(config, task.value) == schema


def test_registry_and_stub_methods_refuse_without_fallback():
    from kumo_core.explain.walk_attribution import (
        METHOD_MODULES,
        UnsupportedWalkAttribution,
        method_module,
    )
    assert set(METHOD_MODULES) == {'hetero_lrp_gamma_reference_v1',
                                   'hetero_gnn_lrp_weight_gamma_v1',
                                   'hetero_gnn_gi_degree_t_v1'}
    for method in METHOD_MODULES:
        module = method_module(method)
        assert module.METHOD_ID == method
        for name in ('qualify', 'make_rule', 'summarize', 'validate_row'):
            assert callable(getattr(module, name))
    for method in ('hetero_gnn_lrp_weight_gamma_v1', 'hetero_gnn_gi_degree_t_v1'):
        stub = method_module(method)
        config = WalkAttributionConfig(method=method)
        for call in (lambda: stub.qualify(fixture(), config),
                     lambda: stub.make_rule(None, None, config),
                     lambda: stub.summarize(None, None, config, None),
                     lambda: stub.validate_row({})):
            with pytest.raises(UnsupportedWalkAttribution,
                               match=f'^{method} is not implemented$'):
                call()
    with pytest.raises(ValueError, match='Unknown'):
        method_module('hetero_lrp_mystery_v1')


def test_inspection_with_a_method_config_has_no_hooks():
    from kumo_core.explain.walk_attribution import inspect_walk_capability
    module = fixture()
    report = inspect_walk_capability(module, config=WalkAttributionConfig())
    assert report['supported'] and report['method'] == 'hetero_lrp_gamma_reference_v1'
    assert 'hetero_gnn' in report['qualification']
    assert report['architecture']['gnn_bias'] and report['architecture']['head_bias']
    stub = inspect_walk_capability(
        module, config=WalkAttributionConfig(method='hetero_gnn_gi_degree_t_v1'))
    assert not stub['supported'] and 'not implemented' in stub['reason']
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in module.modules())
    with pytest.raises(TypeError):
        inspect_walk_capability(module, config={'gamma': 0.0})

