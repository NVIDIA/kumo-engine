"""Independent scalar algebra oracle for the Engine GNN-LRP adapter.

These tests do not derive expected relevance by running the implementation.
The graph has event values 2 and 4, entity value 1, unit message weights,
identity skips, and a readout weight of +/-3.
"""
import pytest
import torch
from torch import nn

from kumo_core.explain.walk_attribution import (WalkAttributionConfig,
                                                WalkCapture,
                                                UnsupportedWalkAttribution)
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead


def fixture(aggregation='sum', depth=1, sign=1, device='cpu'):
    module = nn.Module()
    module.entity_table_name = 'entities'
    relation = ('events', 'entity_id', 'entities')
    module.model = HeteroGNN(channels=1, num_mp_layers=depth,
                             metadata=(['events', 'entities'], [relation]),
                             aggr=aggregation, act='relu', norm=None,
                             bias=False, learn_skip=False)
    module.head = NodePredHead(in_channels=1, out_channels=1, num_layers=1,
                               norm=None, param_keys=[])
    with torch.no_grad():
        for param in module.model.parameters():
            param.fill_(1)
        module.head.mlp.lins[0].weight.fill_(3 * sign)
        module.head.mlp.lins[0].bias.zero_()
    module = module.eval().to(device)
    x = {
        'events': torch.tensor([[2.], [4.]], device=device),
        'entities': torch.tensor([[1.]], device=device)
    }
    edges = {relation: torch.tensor([[0, 1], [0, 0]], device=device)}
    return module, x, edges


def capture(module, x, edges, **kwargs):
    with WalkCapture(module, WalkAttributionConfig(epsilon=1e-12,
                                                   **kwargs)) as recorder:
        output = module.head(module.model(x, edges)['entities'], None)
    return output, recorder.result(seed_index=0)


@pytest.mark.parametrize('aggregation,expected', [('sum', [3., 6., 12.]),
                                                  ('mean', [3., 3., 6.])])
@pytest.mark.parametrize('sign', [1, -1])
def test_manual_one_layer_walks(aggregation, expected, sign):
    module, x, edges = fixture(aggregation=aggregation, sign=sign)
    ordinary = module.head(module.model(x, edges)['entities'], None)
    output, result = capture(module, x, edges)
    torch.testing.assert_close(output, ordinary, rtol=0, atol=0)
    assert float(output[0, 0]) == sum(expected) * sign
    values = sorted(walk['relevance'] for walk in result['walks'])
    assert values == pytest.approx(sorted(value * sign for value in expected),
                                   abs=1e-5)


def test_two_layer_paths_do_not_collapse_repeated_records():
    module, x, edges = fixture(depth=2)
    output, result = capture(module, x, edges)
    assert float(output[0, 0]) == 39.
    assert sorted(w['relevance'] for w in result['walks']) == pytest.approx(
        [3., 6., 6., 12., 12.], abs=1e-5)


def test_positive_gamma_signed_inputs_have_manual_rule_expectation():
    module, x, edges = fixture()
    # Message preactivation is -2 + 4; entity skip contributes 1.
    # gamma=1 transforms local terms to [-2, 8, 2], denominator 8.
    # The readout is 3 * (1 - 2 + 4) = 9.
    # Therefore relevance is [-2.25, 9, 2.25], not gradient * input.
    x['events'][0] = -2
    output, result = capture(module, x, edges, gamma=1.)
    assert float(output.detach()[0, 0]) == 9.
    assert sorted(w['relevance']
                  for w in result['walks']) == pytest.approx([-2.25, 2.25, 9.],
                                                             abs=1e-5)


def test_zero_input_is_successful_zero_relevance():
    module, x, edges = fixture()
    x['events'][0] = 0
    output, result = capture(module, x, edges)
    assert float(output[0, 0]) == 15.
    assert sum(w['relevance'] for w in result['walks']) == pytest.approx(15.)


def test_capture_does_not_rewrite_real_gradients():
    module, x, edges = fixture()
    x['events'].requires_grad_(True)
    output, _ = capture(module, x, edges)
    gradient, = torch.autograd.grad(output.sum(), x['events'])
    torch.testing.assert_close(gradient, torch.tensor([[3.], [3.]]))


def test_capture_removes_hooks_after_exception():
    module, x, edges = fixture()
    before = [
        len(m._forward_hooks) + len(m._forward_pre_hooks)
        for m in module.modules()
    ]
    with pytest.raises(RuntimeError, match='injected'):
        with WalkCapture(module, WalkAttributionConfig()):
            raise RuntimeError('injected')
    after = [
        len(m._forward_hooks) + len(m._forward_pre_hooks)
        for m in module.modules()
    ]
    assert before == after


def test_actual_operator_controls_support_not_config_label():
    from torch_geometric.nn.aggr import MaxAggregation
    module, _, _ = fixture()
    conv = module.model.hetero_convs[0][('events', 'entity_id', 'entities')]
    conv.aggr = MaxAggregation()
    with pytest.raises(UnsupportedWalkAttribution):
        with WalkCapture(module, WalkAttributionConfig()):
            pytest.fail('Unsupported actual aggregator accepted')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU unavailable')
def test_gpu_scalar_oracle():
    module, x, edges = fixture(device='cuda')
    output, result = capture(module, x, edges)
    assert float(output[0, 0]) == 21.
    assert sorted(w['relevance']
                  for w in result['walks']) == pytest.approx([3., 6., 12.])
