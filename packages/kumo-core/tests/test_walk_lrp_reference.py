# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The legacy contribution-gamma rule on the shared enumerator (v2).

On float64 fixtures the v2 rule must reproduce v1 `WalkCapture` walk by
walk. v1 adds its epsilon through `torch.where` with Python scalars, which
yields the float32 value 9.9999997e-10, while v2 uses the exact float64
epsilon; the 1e-12 relative tolerance absorbs that difference.
"""
import math

import pytest
import torch

from kumo_core.explain.walk_attribution import (
    WalkAttributionConfig,
    WalkCapture,
)
from kumo_core.explain.walk_binary import Selection
from kumo_core.explain.walk_enumerate import enumerate_walks
from kumo_core.explain.walk_lrp_reference import (
    ContributionGammaRule,
    GammaLRPRule,
    epsilon_split,
    make_rule,
    summarize,
)
from kumo_core.explain.walk_schema import walk_key
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.modeling.heads.node_prediction import NodePredHead
from tests.test_walk_graph import (
    b1_batch,
    b1_forward,
    b1_module,
    capture,
    preprocessing_module,
    random_b1,
)

F64 = torch.float64


def reference_module(aggregation='sum', depth=1, sign=1.0, signed_input=False):
    relation = ('events', 'entity_id', 'entities')
    module = torch.nn.Module()
    module.entity_table_name = 'entities'
    module.model = HeteroGNN(channels=1, num_mp_layers=depth,
                             metadata=(['events', 'entities'], [relation]),
                             aggr=aggregation, act='relu', norm=None, bias=False,
                             learn_skip=False)
    module.head = NodePredHead(in_channels=1, out_channels=1, num_layers=1, norm=None,
                               param_keys=[])
    module.double().eval()
    with torch.no_grad():
        for param in module.model.parameters():
            param.fill_(1)
        module.head.mlp.lins[0].weight.fill_(3 * sign)
        module.head.mlp.lins[0].bias.zero_()
    x = {'events': torch.tensor([[-2. if signed_input else 2.], [4.]], dtype=F64),
         'entities': torch.tensor([[1.]], dtype=F64)}
    edges = {relation: torch.tensor([[0, 1], [0, 0]])}
    return module, lambda: module.head(module.model(dict(x), dict(edges))['entities'], None)


FIXTURES = {
    'b1': lambda: (lambda m, b: (m, lambda: b1_forward(m, b)))(b1_module(), b1_batch()),
    'b1_biased': lambda: (lambda m, b: (m, lambda: b1_forward(m, b)))(
        b1_module(gnn_bias=True, head_bias=True, head_layers=2), b1_batch()),
    'preprocessing': preprocessing_module,
    'preprocessing_identity': lambda: preprocessing_module(act=None),
    'reference_sum_2': lambda: reference_module('sum', 2),
    'reference_mean_2_negative': lambda: reference_module('mean', 2, -1.0),
    'reference_signed': lambda: reference_module(signed_input=True),
    'random_1': lambda: random_b1(1),
    'random_2': lambda: random_b1(2, pre=1, aggr='sum', learn_skip=False),
    'random_3': lambda: random_b1(3, act=None, head_layers=1),
}


def v1_walks(module, run, config):
    with WalkCapture(module, config) as cap:
        run()
        result = cap.result()
    pre, T = module.model.num_pre_mp_layers, module.model.num_mp_layers
    walks = {}
    for walk in result['walks']:
        key = []
        for step in walk['steps']:
            if not pre <= step['layer'] < pre + T:
                continue  # head and preprocessing steps are not walk steps in v2
            t = step['layer'] - pre + 1
            if step['kind'] == 'relation':
                key.append((t, 'relation', *step['type'], step['edge_occurrence']))
            else:
                key.append((t, 'self_update', step['table'], '', step['table'],
                            step['occurrence']))
        walks[tuple(key)] = walk['relevance']
    return walks, result


def v2_enumeration(module, run, config, sign=1, output_index=0):
    cap, _ = capture(module, run)
    computation = cap.computation()
    selection = Selection('regression', output_index, sign, None)
    rule = make_rule(computation, selection, config)
    return computation, enumerate_walks(computation, rule, max_walks=config.max_walks)


def close(a, b, rel=1e-12):
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


@pytest.mark.parametrize('gamma', [0.0, 0.25, 1.0])
@pytest.mark.parametrize('name', sorted(FIXTURES))
def test_v2_legacy_rule_reproduces_v1_walk_by_walk(name, gamma):
    module, run = FIXTURES[name]()
    config = WalkAttributionConfig(gamma=gamma, top_k=10000, max_walks=100000)
    expected, result = v1_walks(module, run, config)
    computation, enumeration = v2_enumeration(module, run, config)
    actual = {walk_key(w.steps): w.relevance for w in enumeration.walks}
    assert set(actual) == set(expected)
    assert all(close(actual[key], expected[key]) for key in expected)
    assert close(math.fsum(enumeration.bias.values()), result['bias_relevance'])
    assert abs(math.fsum(enumeration.stabilizer.values())
               - result['stabilizer_residual']) <= 1e-12 * max(1.0, abs(result['score']))
    if any(float(p.abs().sum()) > 0 for name, p in module.named_parameters()
           if name.endswith('bias')):
        assert any(enumeration.bias.values())


def test_bias_shares_are_split_by_kind():
    module, run = preprocessing_module()
    config = WalkAttributionConfig(top_k=10000)
    _, enumeration = v2_enumeration(module, run, config)
    assert all(enumeration.bias[kind] != 0.0
               for kind in ('preprocessing', 'message_passing', 'head'))
    _, result = v1_walks(module, run, config)
    assert close(math.fsum(enumeration.bias.values()), result['bias_relevance'])


def b3_module():
    """Appendix B3: h = (1, 2), w = (2, -0.5), z = 1, no biases."""
    module = torch.nn.Module()
    module.entity_table_name = 'entities'
    module.model = HeteroGNN(channels=2, num_mp_layers=1, metadata=(['entities'], []),
                             aggr='sum', act=None, norm=None, bias=False, learn_skip=False)
    module.head = NodePredHead(2, 1, num_layers=1, norm=None, param_keys=[], bias=False)
    module.double().eval()
    with torch.no_grad():
        module.head.mlp.lins[0].weight.copy_(torch.tensor([[2.0, -0.5]]))
    x = {'entities': torch.tensor([[1.0, 2.0]], dtype=F64)}
    return module, lambda: module.head(module.model(dict(x), {})['entities'])


@pytest.mark.parametrize('gamma,positive,folded', [
    (0.25, (5 / 3, -2 / 3), (-8 / 3, 5 / 3)),
    (0.0, (2.0, -1.0), (-2.0, 1.0)),
])
def test_sign_folding_fixture_b3(gamma, positive, folded):
    module, run = b3_module()
    config = WalkAttributionConfig(gamma=gamma, epsilon=1e-12)
    vectors = {}
    for sign in (1, -1):
        _, enumeration = v2_enumeration(module, run, config, sign=sign)
        (walk, ) = enumeration.walks
        vectors[sign] = walk.input_relevance.tolist()
        assert close(walk.relevance, sign * 1.0, rel=1e-9)
    assert all(abs(a - b) <= 1e-9 for a, b in zip(vectors[1], positive))
    assert all(abs(a - b) <= 1e-9 for a, b in zip(vectors[-1], folded))
    negated = [-v for v in vectors[1]]
    if gamma > 0:  # Negating the explanation of z is not the folded rule.
        assert max(abs(a - b) for a, b in zip(negated, vectors[-1])) > 0.9
    else:
        assert all(abs(a - b) <= 1e-9 for a, b in zip(negated, vectors[-1]))


def test_vectors_live_at_h0_and_sum_to_walk_relevance():
    module, run = preprocessing_module()
    computation, enumeration = v2_enumeration(module, run, WalkAttributionConfig(top_k=100))
    for walk in enumeration.walks:
        vector = walk.input_relevance
        assert vector.dtype == F64 and vector.shape == (2, )
        assert close(math.fsum(vector.tolist()), walk.relevance)
        h0 = computation.h0[walk.source[0]][walk.source[1]]
        assert all(float(r) == 0.0 for r, v in zip(vector, h0) if float(v) == 0.0)


def test_identity_root_and_zero_inputs_give_exact_zero_channels():
    module, batch = b1_module(learn_skip=False), b1_batch(
        x={'customer': [[0.]], 'transactions': [[2.], [3.]], 'article': [[4.]]})
    cap, _ = capture(module, lambda: b1_forward(module, batch))
    computation = cap.computation()
    rule = make_rule(computation, Selection('regression', 0, 1, None), WalkAttributionConfig())
    enumeration = enumerate_walks(computation, rule, max_walks=100)
    from_seed = [w for w in enumeration.walks if w.source == ('customer', 0)]
    assert from_seed and all(w.input_relevance.tolist() == [0.0] for w in from_seed)


def test_exact_float64_epsilon_in_v2():
    total = torch.tensor([0.0], dtype=F64)
    children, shares, residual = epsilon_split([torch.zeros(1, 1, dtype=F64)], [], torch.tensor(
        [1.0], dtype=F64), 1e-9)
    assert children[0].tolist() == [0.0] and residual == 1.0
    # v1's `torch.where(total >= 0, 1e-9, -1e-9)` is float32.
    assert float(torch.where(total >= 0, 1e-9, -1e-9)) == pytest.approx(9.9999997e-10, rel=1e-8)
    assert float(torch.where(total >= 0, 1e-9, -1e-9)) != 1e-9


def test_skeleton_is_parameterized_by_the_modified_contribution():
    # A subclass that forms the same contribution reproduces the legacy rule;
    # P3's weight-gamma rule overrides only `modified_contribution`.
    class Same(GammaLRPRule):
        gamma_rule = 'contribution'

        def modified_contribution(self, inputs, weight, denominator):
            z = inputs[:, None] * weight.T / denominator
            return z + self.gamma * z.clamp(min=0)

    module, run = preprocessing_module()
    cap, _ = capture(module, run)
    computation = cap.computation()
    config = WalkAttributionConfig(gamma=0.25, top_k=100)
    selection = Selection('regression', 0, 1, None)
    legacy = enumerate_walks(computation, ContributionGammaRule(computation, selection, config),
                             max_walks=100)
    same = enumerate_walks(computation, Same(computation, selection, config), max_walks=100)
    assert [w.relevance for w in legacy.walks] == [w.relevance for w in same.walks]
    assert legacy.bias == same.bias and legacy.stabilizer == same.stabilizer
    with pytest.raises(NotImplementedError):
        GammaLRPRule(computation, selection, config).modified_contribution(None, None, 1)
    settings = summarize(computation, selection, config, legacy)['method_settings']
    assert settings == {'gamma': 0.25, 'gamma_rule': 'contribution',
                        'bias_rule': 'gamma_on_constant_input', 'epsilon': 1e-9,
                        'stabilizer': 'sign_preserving_epsilon',
                        'identity_root_rule': 'identity_as_weight',
                        'sign_folding': 'selected_output_row'}


def test_selected_output_and_seed_of_a_multiseed_forward():
    # Two seeds and a two-output head: v2 explains output 1 of seed 1 as v1.
    module, _ = reference_module()
    module.head = NodePredHead(1, 2, num_layers=1, norm=None, param_keys=[]).double().eval()
    with torch.no_grad():
        module.head.mlp.lins[0].weight.copy_(torch.tensor([[3.], [-3.]]))
        module.head.mlp.lins[0].bias.fill_(4)
    relation = ('events', 'entity_id', 'entities')
    x = {'events': torch.tensor([[2.], [4.], [1.]], dtype=F64),
         'entities': torch.tensor([[1.], [5.]], dtype=F64)}
    edges = {relation: torch.tensor([[0, 1, 2], [0, 0, 1]])}

    def run():
        return module.head(module.model(dict(x), dict(edges))['entities'], None)

    config = WalkAttributionConfig(output_index=1, top_k=100)
    with WalkCapture(module, config) as cap:
        run()
        v1 = cap.result(seed_index=1)
    graph_capture, _ = capture(module, run)
    computation = graph_capture.computation(seed_index=1)
    rule = make_rule(computation, Selection('regression', 1, 1, None), config)
    enumeration = enumerate_walks(computation, rule, max_walks=100)
    assert computation.seed == ('entities', 1)
    assert sorted(w.relevance for w in enumeration.walks) == pytest.approx(
        sorted(w['relevance'] for w in v1['walks']), rel=1e-12)
    assert close(math.fsum(enumeration.bias.values()), v1['bias_relevance'])
    assert v1['score'] == float(computation.head_output[1]) == -3 * 6 + 4
