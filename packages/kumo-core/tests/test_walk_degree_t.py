# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Degree-T (GNN-GI) method: qualification, oracles, exactness and rows.

Numeric expectations come from hand values (fixture B1) or from the
independent `walk_degree_t_oracles` module, which never imports a walk
module. Oracle comparisons use interior points (every relevant gated
pre-activation at least 1e-2 from zero); kinks are tested only by
dedicated exact-zero fixtures.
"""
import ast
from collections import Counter
import copy
import dataclasses
import functools
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from kumoapi.encoder import Index
from kumoapi.task import TaskType
from kumoapi.typing import ColStatType

from kumo_core.encoders.target.base import TargetEncoder
from kumo_core.explain import walk_degree_t as gi
from kumo_core.explain.perturbation import RecordRemovalPlan, remove_record
from kumo_core.explain.walk_attribution import (
    DEGREE_T_METHOD,
    UnsupportedWalkAttribution,
    WalkAttributionConfig,
    inspect_walk_capability,
    method_module,
)
from kumo_core.explain.walk_binary import (
    Selection,
    output_block,
    resolve_selection,
)
from kumo_core.explain.walk_enumerate import enumerate_walks
from kumo_core.explain.walk_graph import (
    STRUCTURAL_CODES,
    GraphCapture,
    capture_numerics,
    float32_matmul_precision,
    mkldnn_matmul_fp32_precision,
)
from kumo_core.explain.walk_schema import (
    WalkSchemaError,
    build_row,
    explain_capture,
    row_numerics,
    tolerance,
    validate_v2_row,
    walk_key,
)
from kumo_core.modeling.calibration import PlatScaling
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.train.loss import BinaryCrossEntropyLoss
from tests.walk_degree_t_oracles import (
    ART_FK,
    ART_REV,
    F64,
    FK,
    REV,
    Fixture,
    GatedOracle,
    as_float64,
    b1,
    interior_seed,
    precision_relhm,
    relhm,
    scoring_module,
    t3_trimmed,
    t3_untrimmed,
)

ID, SAMPLE_ID = 'a' * 64, 'b' * 64
HIGHEST = {'precision_mode': 'highest', 'mkldnn_matmul_fp32_precision': 'ieee',
           'num_threads': 2, 'cpu_capability': 'AVX512', 'torch_version': '2.13.0+cpu'}
CONFIG = WalkAttributionConfig(method=DEGREE_T_METHOD, feature_vectors=True, bags=True,
                               top_k=100000, max_walks=100000)
#: Interior fixture builders (the seed is found by the oracle's margin search).
BUILDERS = {
    'relhm': relhm,
    't3_trimmed': t3_trimmed,
    't3_untrimmed_sum': t3_untrimmed,
    't3_untrimmed_mean': lambda seed: t3_untrimmed(seed, aggr='mean', act='relu',
                                                   signed=False),
}
FIXTURES = ['b1'] + sorted(BUILDERS)


@functools.lru_cache(maxsize=None)
def seed_of(name):
    return interior_seed(BUILDERS[name])


def fixture(name):
    """A fresh deterministic fixture (tests may mutate it)."""
    return b1() if name == 'b1' else BUILDERS[name](seed_of(name))


def no_hooks(module):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in module.modules())


def capture(fx, config=CONFIG, mode='highest'):
    with float32_matmul_precision(mode), GraphCapture(fx.module, config) as cap:
        logits = fx.forward()
    return cap, logits


def computation_of(fx, config=CONFIG, mode='highest'):
    cap, logits = capture(fx, config, mode)
    numerics = row_numerics(HIGHEST if mode == 'highest' else {'precision_mode': mode},
                            str(logits.dtype).removeprefix('torch.'))
    return cap.computation(numerics=numerics)


def select(sign=1):
    return SimpleNamespace(output_index=0, logit_sign=sign)


def run_gi(computation, sign=1, max_walks=100000):
    rule = gi.make_rule(computation, select(sign), CONFIG)
    enumeration = enumerate_walks(computation, rule, max_walks=max_walks)
    return enumeration, gi.summarize(computation, select(sign), CONFIG, enumeration)


def scores_of(enumeration):
    return {walk_key(walk.steps): walk.relevance for walk in enumeration.walks}


def make_binary(fx, calibration=(0.8, -0.1)):
    """Attach BCE, a fitted two-label target encoder (`True` is encoded class
    1) and a fitted Platt calibrator, as a binary scoring module has them."""
    module = fx.module
    dtype = next(module.parameters()).dtype
    module.task_type = TaskType.BINARY_CLASSIFICATION
    module.loss = BinaryCrossEntropyLoss(TaskType.BINARY_CLASSIFICATION)
    index = Index()
    index._stats = {ColStatType.CATEGORY_COUNTS: [[True, 123], [False, 37]]}
    module.target_encoder = TargetEncoder.from_encoder(TaskType.BINARY_CLASSIFICATION, index)
    module.calibrator = PlatScaling(TaskType.BINARY_CLASSIFICATION)
    with torch.no_grad():
        module.calibrator.weight_and_bias.copy_(torch.tensor([[calibration[0]],
                                                              [calibration[1]]]))
    module.calibrator.set_extra_state({'is_fitted': True})
    module.calibrator.to(dtype)
    module.eval()
    return fx


def _explain(fx, config, mode, numerics):
    cap, logits = capture(fx, config, mode)
    module = fx.module
    if getattr(module, 'task_type', None) == TaskType.BINARY_CLASSIFICATION:
        selection = resolve_selection(module, config)
    else:
        selection = Selection('regression', config.output_index, 1, None)
    output = output_block(selection, module, logits[0])
    if numerics is None:
        with float32_matmul_precision(mode):
            numerics = capture_numerics()
    numerics = row_numerics(numerics, str(logits.dtype).removeprefix('torch.'))
    row = explain_capture(cap, config=config, selection=selection, output=output,
                          sample=fx.sample(), origin='original_forward_attribution',
                          prediction_id=ID, sample_sha256=SAMPLE_ID, numerics=numerics)
    return row, cap, numerics


def row_for(fx, config=CONFIG, *, mode='highest', numerics=None):
    """One v2 row through P2's public pipeline (qualify, capture, build, validate)."""
    return _explain(fx, config, mode, numerics)[0]


def explained(fx, config=CONFIG, *, mode='highest'):
    """`row_for` and the computation of the same capture (rows that build)."""
    row, cap, numerics = _explain(fx, config, mode, None)
    return row, cap.computation(numerics=numerics)


def close(actual, expected, rel=1e-12):
    return abs(actual - expected) <= rel * max(1.0, abs(expected))


def module_value(fx):
    return float(fx.forward().detach().reshape(-1)[0])


# Registration and qualification ##############################################

def test_method_registration_and_configuration():
    assert gi.METHOD_ID == DEGREE_T_METHOD == 'hetero_gnn_gi_degree_t_v1'
    assert method_module(DEGREE_T_METHOD) is gi
    for kwargs in ({'gamma': 0.25}, {'epsilon': 1e-6}):
        with pytest.raises(ValueError, match='no gamma rule'):
            WalkAttributionConfig(method=DEGREE_T_METHOD, **kwargs)
    with pytest.raises(ValueError, match='bags=True requires'):
        WalkAttributionConfig(bags=True)
    module = b1().module
    for config in (WalkAttributionConfig(), None):
        with pytest.raises(UnsupportedWalkAttribution, match='^config_not_gnn_gi: '):
            gi.qualify(module, config)
    assert gi.qualify(module, CONFIG) == list(STRUCTURAL_CODES) + ['bias_free_gnn',
                                                                    'bias_free_head']
    assert set(gi.CLAIM_CODES) >= set(STRUCTURAL_CODES) | {'single_edge_gathers'}


def _head_flag_without_parameters():
    module = scoring_module(3, 2, head_bias=True)
    for lin in module.head.mlp.lins:  # as P2's binary_scoring_module builds its head
        lin.register_parameter('bias', None)
    return module


def _with_encoder(module):
    module.encoder = torch.nn.Linear(3, 3).double().eval()  # encoder biases: fixed H_0
    return module


def _dropout(module):
    for post in module.model.post_convs:
        for ops in post.layer_dict.values():
            ops[-1].p = 0.5
    return module


ACCEPTED = {
    'mean_learned_root': lambda: scoring_module(3, 2),
    'sum_identity_root': lambda: scoring_module(3, 2, aggr='sum', learn_skip=False),
    'identity_activation': lambda: scoring_module(3, 2, act=None),
    'preprocessing_and_two_layer_head': lambda: scoring_module(3, 2, pre=1, head_layers=2),
    'one_layer': lambda: scoring_module(3, 1),
    'three_layers': lambda: scoring_module(3, 3),
    'encoder_with_biases': lambda: _with_encoder(scoring_module(3, 2)),
    'eval_mode_dropout': lambda: _dropout(scoring_module(3, 2)),
    'head_flag_without_bias_parameters': _head_flag_without_parameters,
}


@pytest.mark.parametrize('name', sorted(ACCEPTED))
def test_qualify_accepts_the_bias_free_profile(name):
    module = ACCEPTED[name]()
    codes = gi.qualify(module, CONFIG)
    assert codes == list(STRUCTURAL_CODES) + list(gi.QUALIFY_CODES)
    report = inspect_walk_capability(module, config=CONFIG)
    assert report['supported'] and report['qualification'] == codes
    assert not report['architecture']['gnn_bias'] and not report['architecture']['head_bias']
    assert no_hooks(module)


def _attach_bias(lin):
    lin.bias = torch.nn.Parameter(torch.zeros(lin.out_features, dtype=F64))


def _conv(module, layer=0, relation=FK):
    return module.model.hetero_convs[layer][relation]


def _relu_ops(module):
    return [op for post in module.model.post_convs for ops in post.layer_dict.values()
            for op in ops if isinstance(op, torch.nn.ReLU)]


def _mutated(build, change):
    def make():
        module = build()
        change(module)
        return module
    return make


def _subclass(instance, base):
    instance.__class__ = type(f'Other{base.__name__}', (base, ), {})


def _base():
    return scoring_module(3, 2)


def _head(**options):
    return _mutated(_base, lambda m: setattr(m, 'head', NodePredHead(
        3, 1, **{'num_layers': 2, 'norm': None, 'param_keys': [], 'bias': False,
                 **options}).double().eval()))


def _time_encoder_without_flag(module):
    module.head = NodePredHead(3, 1, norm=None, param_keys=[], bias=False,
                               prediction_time_encodings=True).double().eval()
    module.head.prediction_time_encodings = False


def _drop_relu(module):
    for post in module.model.post_convs:
        for ops in post.layer_dict.values():
            del ops[0]


REJECTIONS = {
    'gnn_bias': (lambda: scoring_module(3, 2, gnn_bias=True), '^gnn_bias_present: '),
    # Identity roots: only the relation convolutions carry biases.
    'conv_bias_only': (lambda: scoring_module(3, 2, gnn_bias=True, learn_skip=False),
                       '^gnn_bias_present: relation'),
    'root_bias': (_mutated(_base, lambda m: _attach_bias(
        m.model.root_skips[1].lin_dict['transactions'])), '^gnn_bias_present: root skip'),
    'preprocessing_bias': (_mutated(lambda: scoring_module(3, 2, pre=1), lambda m: _attach_bias(
        m.model.lins[0].lin_dict['article'])), '^gnn_bias_present: preprocessing'),
    'head_bias': (lambda: scoring_module(3, 2, head_bias=True), '^head_bias_present: '),
    'falsy_conv_flag_with_bias': (_mutated(_base, lambda m: _attach_bias(_conv(m).lin)),
                                  '^inconsistent_bias_configuration: '),
    'truthy_conv_flag_without_bias': (_mutated(_base, lambda m: setattr(_conv(m), 'bias', True)),
                                      '^inconsistent_bias_configuration: '),
    'gelu': (lambda: scoring_module(3, 2, act='gelu'), 'ReLU/identity'),
    'elu': (lambda: scoring_module(3, 2, act='elu'), 'ReLU/identity'),
    'leaky_relu': (lambda: scoring_module(3, 2, act='leaky_relu'), 'ReLU/identity'),
    'leaky_relu_head': (_head(act='leaky_relu'), 'ReLU/identity'),
    'gelu_head': (_head(act='gelu'), 'ReLU/identity'),
    'layer_norm': (_mutated(_base, lambda m: setattr(m.model, 'norm', 'layer_norm')),
                   'no normalization'),
    'batch_norm_head': (_head(norm='batch_norm'), 'no normalization'),
    'layer_norm_head': (_head(norm='layer_norm'), 'no normalization'),
    'max_aggregation': (lambda: scoring_module(3, 2, aggr='max'), 'sum or mean'),
    'min_aggregation': (lambda: scoring_module(3, 2, aggr='min'), 'sum or mean'),
    'multi_aggregation': (_mutated(_base, lambda m: setattr(m.model, 'aggr', ['sum', 'mean'])),
                          'sum or mean'),
    'prediction_time_encodings': (_head(prediction_time_encodings=True), 'Time/distribution'),
    'time_encoder_without_flag': (_mutated(_base, _time_encoder_without_flag),
                                  '^prediction_time_encoding: '),
    'distribution_head': (_head(param_keys=['loc', 'scale']), 'Time/distribution'),
    'training_model': (_mutated(_base, lambda m: m.model.train()), 'evaluation mode'),
    'training_head': (_mutated(_base, lambda m: m.head.train()), 'evaluation mode'),
    'training_encoder': (_mutated(lambda: _with_encoder(_base()), lambda m: m.encoder.train()),
                         'evaluation mode'),
    'compiled_model': (_mutated(_base, lambda m: setattr(m, 'model', torch.compile(m.model))),
                       'Requires HeteroGNN'),
    'extra_parameter': (_mutated(_base, lambda m: m.model.register_parameter(
        'extra', torch.nn.Parameter(torch.zeros(1, dtype=F64)))),
        '^unregistered_parameter: model.extra'),
    'unexpected_buffer': (_mutated(_base, lambda m: m.head.register_buffer(
        'scale', torch.ones(1, dtype=F64))), '^unexpected_buffer: head.scale'),
    'plain_last_false': (_mutated(_base, lambda m: setattr(m.head.mlp, 'plain_last', False)),
                         'plain linear'),
    'act_string_without_module': (_mutated(_base, lambda m: setattr(m.model, 'act', None)),
                                  '^activation: '),
    'relu_string_without_module': (_mutated(_base, _drop_relu), '^activation: '),
    'readout_act_disagrees': (_mutated(_base, lambda m: setattr(m.head, 'act', None)),
                              '^activation: '),
    'inplace_relu': (_mutated(_base, lambda m: setattr(_relu_ops(m)[0], 'inplace', True)),
                     '^unregistered_operator: '),
    'no_message_passing': (lambda: scoring_module(3, 0), '^unsupported_model: '),
    'subclassed_root': (_mutated(_base, lambda m: _subclass(m.model.root_skips[0],
                                                            type(m.model.root_skips[0]))),
                        'Unregistered root operator'),
    'subclassed_relation_linear': (_mutated(_base, lambda m: _subclass(_conv(m).lin,
                                                                       torch.nn.Linear)),
                                   '^unregistered_operator: relation'),
}


@pytest.mark.parametrize('name', sorted(REJECTIONS))
def test_qualify_rejects_before_hooks_without_mutation(name):
    build, match = REJECTIONS[name]
    module = build()
    state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    training = [m.training for m in module.modules()]
    with pytest.raises(UnsupportedWalkAttribution, match=match):
        gi.qualify(module, CONFIG)
    with pytest.raises(UnsupportedWalkAttribution, match=match):
        with GraphCapture(module, CONFIG):
            pass
    report = inspect_walk_capability(module, config=CONFIG)
    assert not report['supported']
    assert no_hooks(module)
    assert [m.training for m in module.modules()] == training
    after = module.state_dict()
    assert set(after) == set(state) and all(torch.equal(after[k], v) for k, v in state.items())


def test_biases_in_a_capture_qualified_for_another_method_make_the_row_unsupported():
    fx = b1(gnn_bias=True, head_bias=True)
    with GraphCapture(fx.module, WalkAttributionConfig()) as cap:  # LRP qualification
        logits = fx.forward()
    selection = Selection('regression', 0, 1, None)
    row = explain_capture(cap, config=CONFIG, selection=selection,
                          output=output_block(selection, fx.module, logits[0]),
                          sample=fx.sample(), origin='original_forward_attribution',
                          prediction_id=ID, sample_sha256=SAMPLE_ID,
                          numerics=row_numerics(HIGHEST, 'float64'))
    assert row['status'] == 'unsupported' and row['reason'].startswith('biases_present: ')
    validate_v2_row(row)


def test_a_claim_needs_the_gnn_gi_qualification_of_the_capture():
    # A bias-free capture qualified for the LRP method lacks the GNN-GI
    # parameter checks, so its degree-T claim cannot validate: fail closed.
    fx = b1()
    with GraphCapture(fx.module, WalkAttributionConfig()) as cap:
        logits = fx.forward()
    selection = Selection('regression', 0, 1, None)
    row = explain_capture(cap, config=CONFIG, selection=selection,
                          output=output_block(selection, fx.module, logits[0]),
                          sample=fx.sample(), origin='original_forward_attribution',
                          prediction_id=ID, sample_sha256=SAMPLE_ID,
                          numerics=row_numerics(HIGHEST, 'float64'))
    assert row['status'] == 'failed' and 'bias_free_gnn' in row['reason']


# Fixture B1 by hand ##########################################################

def _self(layer, table, occurrence):
    return (layer, 'self_update', table, '', table, occurrence)


def _rel(layer, relation, edge):
    return (layer, 'relation', *relation, edge)


#: Contract 6.4 and P0 Appendix B1: the eight walk scores (sum 9).
B1_HAND = {
    (_self(1, 'customer', 0), _self(2, 'customer', 0)): 1.0,
    (_rel(1, FK, 0), _self(2, 'customer', 0)): 1.0,          # t0 = 2, mean of 2
    (_rel(1, FK, 1), _self(2, 'customer', 0)): 1.5,          # t1 = 3
    (_self(1, 'transactions', 0), _rel(2, FK, 0)): 1.0,
    (_rel(1, REV, 0), _rel(2, FK, 0)): 0.5,                  # c = 1
    (_rel(1, ART_REV, 0), _rel(2, FK, 0)): 2.0,              # a = 4
    (_self(1, 'transactions', 1), _rel(2, FK, 1)): 1.5,
    (_rel(1, REV, 1), _rel(2, FK, 1)): 0.5,
}


def _removed_t0(fx):
    """Engine's structural removal of transaction t0 (recomputes means)."""
    pkeys = fx.sample()['nodes']
    nodes = {table: {'pkeys': pkeys[table]['pkeys'], 'num_nodes': x.size(0),
                     'num_sampled_nodes': fx.nodes[table], 'feature_tensors': {}}
             for table, x in fx.x.items()}
    edges = [{'type': list(rel), 'edge_index': index.tolist(),
              'num_sampled_edges': fx.hops[rel]} for rel, index in fx.edges.items()]
    sample = {'seed_table': 'customer', 'train_table_name': 'task',
              'entity_table_name': 'customer', 'nodes': nodes, 'edges': edges}
    group = {'table': 'transactions', 'pkey': pkeys['transactions']['pkeys'][0]}
    removed = remove_record(sample, group, RecordRemovalPlan(
        groups=(group, ), removable_tables=('transactions', ), foreign_keys=(FK, ART_FK)))
    x = dict(fx.x, transactions=fx.x['transactions'][1:])
    edges = {tuple(e['type']): torch.tensor(e['edge_index'], dtype=torch.long).reshape(2, -1)
             for e in removed['edges']}
    return Fixture(fx.module, x, edges,
                   {t: n['num_sampled_nodes'] for t, n in removed['nodes'].items()},
                   {tuple(e['type']): e['num_sampled_edges'] for e in removed['edges']})


def test_b1_hand_values_bag_homogeneity_and_gates_versus_deletion():
    fx = b1()
    row = row_for(fx)
    assert row['status'] == 'available', row.get('reason')
    computation = computation_of(fx)
    enumeration, summary = run_gi(computation)
    assert scores_of(enumeration) == B1_HAND
    assert math.fsum(B1_HAND.values()) == 9.0 == summary['diagnostics']['f_piece']
    assert row['exactness']['decomposition'] == 'exact_degree_t' and row['score'] == 9.0
    assert row['ledger']['precision_residual'] == 0.0
    repeated = [bag for bag in row['bags'] if any(v['multiplicity'] == 2
                                                  for v in bag['variables'])]
    assert len(row['bags']) == 8 and len(repeated) == 1
    assert repeated[0]['relevance'] == 1.0 and repeated[0]['walk_count'] == 1
    assert repeated[0]['variables'] == [{'kind': 'self_update', 'table': 'customer',
                                         'occurrence': 0, 'multiplicity': 2}]
    oracle = GatedOracle(fx)
    assert oracle.evaluate(oracle.gates()) == 9.0
    assert oracle.evaluate(oracle.gates(2.0)) == 36.0
    assert oracle.evaluate(oracle.gates(2.0), gate_weighted=True) == 20.0
    # Gating the t0 -> c edge at both layers keeps every denominator frozen.
    lam = oracle.gates()
    lam[(1, 'rel', FK)][0] = 0.0
    lam[(2, 'rel', FK)][0] = 0.0
    gated = oracle.evaluate(lam)
    through = math.fsum(score for key, score in B1_HAND.items()
                        if any(step[1] == 'relation' and step[2:5] == FK and step[5] == 0
                               for step in key))
    assert gated == 4.5 == 9.0 - through
    # Engine's record removal deletes t0 and recomputes the customer mean.
    assert module_value(_removed_t0(fx)) == 8.0


# Independent oracles ##########################################################

@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('name', FIXTURES)
def test_walk_scores_equal_nested_autograd_mixed_partials(name, sign):
    fx = fixture(name)
    oracle = GatedOracle(fx)
    value = module_value(fx)
    assert close(oracle.evaluate(oracle.gates()), value)  # anchor
    computation = computation_of(fx)
    enumeration, summary = run_gi(computation, sign)
    mine, expected = scores_of(enumeration), oracle.walk_scores(sign=sign)
    assert set(mine) == set(expected) and len(mine) == len(enumeration.walks)
    assert all(close(mine[key], expected[key], 1e-10) for key in expected)
    mass = math.fsum(abs(score) for score in mine.values())
    f_piece = summary['diagnostics']['f_piece']
    assert abs(math.fsum(mine.values()) - f_piece) <= 1e-12 * max(1.0, mass)
    assert close(f_piece, sign * value)
    assert summary['exactness']['reasons'] == [] and summary['exactness']['degree_t_claim']
    if computation.activation is not None:
        assert summary['exactness']['min_abs_gated_preactivation'] >= 1e-2


@pytest.mark.parametrize('name', FIXTURES)
def test_walk_polynomial_equals_the_fixed_mask_forward_at_any_gates(name):
    # With the captured masks fixed, f is exactly sum_W R_W prod_t lambda[t, v_t]
    # for every gate vector, which checks all coefficients jointly.
    fx = fixture(name)
    oracle = GatedOracle(fx)
    masks = oracle.masks_at_one()
    scores = scores_of(run_gi(computation_of(fx))[0])
    generator = torch.Generator().manual_seed(5)
    for _ in range(3):
        lam = {key: 2 * torch.rand(value.shape, generator=generator, dtype=F64)
               for key, value in oracle.gates().items()}
        expected = oracle.evaluate(lam, masks=masks)
        assert close(oracle.polynomial(scores, lam), expected, 1e-10)


def _oracle_variables(bag):
    variables = Counter()
    for v in bag['variables']:
        key = (('rel', tuple(v['relation']), v['edge_occurrence']) if v['kind'] == 'relation'
               else ('self', v['table'], v['occurrence']))
        variables[key] = v['multiplicity']
    return variables


@pytest.mark.parametrize('name,sign', [(name, 1) for name in FIXTURES] + [('t3_trimmed', -1)])
def test_bags_equal_tied_gate_derivatives_over_alpha_factorial(name, sign):
    fx = fixture(name)
    if sign == -1:
        fx = make_binary(fx)
    config = WalkAttributionConfig(**{**CONFIG.to_dict(), 'selected_class': False}) if (
        sign == -1) else CONFIG
    row = row_for(fx, config)
    assert row['status'] == 'available' and row['coverage']['bags'] == 'complete'
    assert row['output']['binary'] is None or row['output']['binary']['logit_sign'] == sign
    oracle = GatedOracle(fx)
    alpha_factorials = []
    for bag in row['bags']:
        variables = _oracle_variables(bag)
        assert close(bag['relevance'], oracle.bag_score(variables, sign=sign), 1e-10)
        alpha_factorials.append(math.prod(math.factorial(m) for m in variables.values()))
    f_piece = row['score'] - row['ledger']['precision_residual']
    assert abs(math.fsum(b['relevance'] for b in row['bags']) - f_piece) <= 1e-12 * max(
        1.0, math.fsum(abs(b['relevance']) for b in row['bags']))
    if row['architecture']['message_passing_layers'] == 3:
        assert any(b['walk_count'] >= 2 for b in row['bags'])  # multi-walk bags
        assert any(f > 1 for f in alpha_factorials)            # repeated variables
        # 1/alpha! belongs to the derivative form only: applied to walk sums it
        # breaks conservation.
        divided = math.fsum(b['relevance'] / f for b, f in zip(row['bags'], alpha_factorials))
        assert abs(divided - f_piece) > 1e-6


@pytest.mark.parametrize('name', FIXTURES)
def test_positive_homogeneity_jointly_and_per_layer(name):
    fx = fixture(name)
    oracle = GatedOracle(fx)
    T = oracle.T
    base = oracle.evaluate(oracle.gates())
    scores = scores_of(run_gi(computation_of(fx))[0])
    for q in (0.5, 2.0):
        joint = oracle.evaluate(oracle.gates(q))
        assert close(joint, q ** T * base)
        assert close(oracle.polynomial(scores, oracle.gates(q)), joint, 1e-10)
        for t in range(1, T + 1):
            assert close(oracle.evaluate(oracle.gates(scale={t: q})), q * base)
        if name != 't3_untrimmed_sum':  # every other fixture has a mean of in-degree >= 2
            weighted = oracle.evaluate(oracle.gates(q), gate_weighted=True)
            assert abs(weighted - q ** T * base) > 1e-6


def test_homogeneity_fails_with_biases_and_the_oracle_reads_engine_biases():
    fx = relhm(seed_of('relhm'), gnn_bias=True, head_bias=True)
    with pytest.raises(UnsupportedWalkAttribution, match='^gnn_bias_present: '):
        gi.qualify(fx.module, CONFIG)
    oracle = GatedOracle(fx)
    base = oracle.evaluate(oracle.gates(), biases=True)
    assert close(base, module_value(fx))  # bias placement matches Engine
    for q in (0.5, 2.0):
        assert abs(oracle.evaluate(oracle.gates(q), biases=True) - q ** 2 * base) > 1e-6


@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('name', FIXTURES)
def test_first_order_oracle_source_vectors_equal_grad_times_input(name, sign):
    fx = fixture(name)
    config = CONFIG
    if sign == -1:
        fx = make_binary(fx)
        config = WalkAttributionConfig(**{**CONFIG.to_dict(), 'selected_class': False})
    row = row_for(fx, config)
    assert row['status'] == 'available', row.get('reason')
    expected = GatedOracle(fx).input_gradient_times_input(sign=sign)
    seen = set()
    for source in row['sources']:
        key = (source['table'], source['occurrence'])
        seen.add(key)
        reference = expected[key[0]][key[1]]
        scale = max(1.0, float(reference.abs().sum()))
        assert all(abs(a - float(b)) <= 1e-12 * scale
                   for a, b in zip(source['input_relevance'], reference))
        assert close(source['relevance'], float(reference.sum()))
    for table, values in expected.items():
        for occurrence in range(values.size(0)):
            if (table, occurrence) not in seen:
                assert float(values[occurrence].abs().max()) == 0.0


@pytest.mark.parametrize('name', ['b1', 'relhm', 't3_trimmed'])
def test_budget_prefix_masses_equal_the_omitted_complete_walks(name):
    computation = computation_of(fixture(name))
    full, summary = run_gi(computation)
    f_piece = summary['diagnostics']['f_piece']
    complete = scores_of(full)
    mass = math.fsum(abs(v) for v in complete.values())
    tol = 1e-12 * max(1.0, mass)
    for budget in range(1, full.expanded_routes):
        part, _ = run_gi(computation, max_walks=budget)
        assert part.prefixes and part.expanded_routes <= budget
        found = scores_of(part)
        assert all(found[key] == complete[key] for key in found)  # bitwise
        masses = [prefix.mass for prefix in part.prefixes]
        assert abs(math.fsum(list(found.values()) + masses) - f_piece) <= tol
        omitted = [key for key in complete if key not in found]
        assert abs(math.fsum(complete[k] for k in omitted) - math.fsum(masses)) <= tol
        for prefix in part.prefixes:
            # A walk completes the prefix iff its steps after the prefix layer
            # are the prefix's expanded suffix.
            suffix = walk_key(prefix.steps)
            extending = [complete[key] for key in omitted if key[prefix.layer:] == suffix]
            assert extending and abs(math.fsum(extending) - prefix.mass) <= tol
        assert math.fsum(abs(m) for m in masses) <= math.fsum(
            abs(complete[k]) for k in omitted) + tol


def test_budget_exhaustion_rows_are_incomplete_without_claim_or_bags():
    fx = fixture('relhm')
    config = WalkAttributionConfig(**{**CONFIG.to_dict(), 'max_walks': 20, 'top_k': 5})
    row = row_for(fx, config)
    assert row['status'] == 'incomplete' and row['coverage']['status'] == 'budget_exhausted'
    assert row['exactness']['reasons'][0] == 'enumeration_incomplete'
    assert not row['exactness']['degree_t_claim']
    assert row['bags'] is None and row['coverage']['bags'] == 'omitted_incomplete_enumeration'
    assert 'complete_enumeration' not in row['exactness']['qualification']
    complete = row_for(fixture('relhm'))
    omitted = complete['score'] - complete['ledger']['precision_residual'] - (
        row['ledger']['displayed_walk_relevance'] + row['ledger']['omitted_display_relevance'])
    assert abs(row['ledger']['omitted_search_relevance'] - omitted) <= 1e-12
    validate_v2_row(row, fx.sample())


# Structure: trimming, empty neighborhoods and gathers #########################

def test_trimming_empty_neighborhoods_and_single_edge_gathers():
    fx = fixture('relhm')
    computation = computation_of(fx)
    assert computation.destinations(2) == {'customer': 1, 'transactions': 0, 'article': 0}
    seed_routes = computation.incoming(2, ('customer', 0)).routes
    # Layer 2 keeps only the hop-1 customer edges; every rev_ edge is trimmed.
    assert [r.kind for r in seed_routes] == ['self_update'] + ['relation'] * 8
    assert {r.relation for r in seed_routes[1:]} == {FK}
    assert {r.denominator for r in seed_routes[1:]} == {8}
    for occurrence in range(8):
        routes = computation.incoming(1, ('transactions', occurrence)).routes
        assert [(r.relation, r.operator, r.denominator) for r in routes] == [
            (None, 'linear', 1), (ART_REV, 'gather', 1), (REV, 'gather', 1)]
    enumeration, summary = run_gi(computation)
    assert len(enumeration.walks) == 1 + 4 * 8 and enumeration.expanded_routes == 2 + 5 * 8
    assert summary['exactness']['relevant_gated_neurons'] == 4 * (1 + 1 + 8)
    # Empty relations add nothing: no route, no bias.
    assert all(not computation.incoming(t, node).biases for t in (1, 2)
               for node in computation.relevant_nodes(t))
    assert not [r for t in (1, 2) for node in computation.relevant_nodes(t)
                for r in computation.incoming(t, node).routes if r.relation == ART_FK]
    # B1: t1 has no article (an empty gather neighborhood contributes 0).
    b1_routes = computation_of(b1()).incoming(1, ('transactions', 1)).routes
    assert [r.relation for r in b1_routes] == [None, REV]
    # A mean destination without incoming edges keeps only its self update.
    untrimmed = computation_of(fixture('t3_untrimmed_mean'))
    assert [r.kind for r in untrimmed.incoming(1, ('article', 1)).routes] == ['self_update']


def test_a_duplicate_gather_makes_the_row_unsupported():
    fx = b1(x={'customer': [[1.]], 'transactions': [[2.], [3.]], 'article': [[4.], [5.]]},
            edges={FK: [[0, 1], [0, 0]], REV: [[0, 0], [0, 1]], ART_REV: [[0, 1], [0, 0]],
                   ART_FK: [[], []]},
            nodes={'customer': [1, 0, 0], 'transactions': [0, 2, 0], 'article': [0, 0, 2]},
            hops={FK: [2, 0], REV: [0, 2], ART_REV: [0, 2], ART_FK: [0, 0]})
    row = row_for(fx)
    assert row['status'] == 'unsupported' and 'multiple incoming' in row['reason']
    assert no_hooks(fx.module)
    validate_v2_row(row)


# Exactness: kinks, disagreements, residuals ###################################

def test_exact_zero_kink_is_a_boundary_without_claim():
    # Layer-1 customer root weight -2.5: a1(c) = -2.5 + (2 + 3) / 2 = 0 exactly
    # while its incoming sources are nonzero (review finding m3).
    fx = b1(values={'model.root_skips.0.lin_dict.customer.weight': [[-2.5]]})
    row = row_for(fx)
    exactness = row['exactness']
    assert row['status'] == 'available'
    assert exactness['piece'] == 'boundary' and exactness['boundary_neurons'] == 1
    assert exactness['reasons'] == ['boundary_piece'] and not exactness['degree_t_claim']
    assert exactness['decomposition'] == 'piecewise_gnn_gi'
    assert exactness['mask_disagreements'] == 0 and exactness['min_abs_gated_preactivation'] == 0
    enumeration, summary = run_gi(computation_of(fx))
    mine = scores_of(enumeration)
    off_piece = {key: (0.0 if key[1] == _self(2, 'customer', 0) else value)
                 for key, value in B1_HAND.items()}
    assert mine == off_piece and math.fsum(mine.values()) == 5.5 == module_value(fx)
    # Both adjacent pieces are valid at gates 1 and conserve, yet split differently.
    oracle = GatedOracle(fx)
    masks = oracle.masks_at_one()
    assert float(masks[('mp', 1, 'customer')][0, 0]) == 0.0  # relu'(0) = 0, as captured
    on = copy.deepcopy(masks)
    on[('mp', 1, 'customer')][0, 0] = 1.0
    off_scores, on_scores = oracle.walk_scores(masks=masks), oracle.walk_scores(masks=on)
    assert off_scores == mine
    assert math.fsum(on_scores.values()) == 5.5 and on_scores != off_scores


def test_a_zero_with_only_zero_sources_is_interior():
    # t1 has H_0 = 0 and only its self update at layer 1: a1(t1) = 0 for every
    # gate value near 1, so either mask gives the same walk scores.
    fx = b1(x={'customer': [[1.]], 'transactions': [[2.], [0.]], 'article': [[4.]]},
            edges={FK: [[0, 1], [0, 0]], REV: [[0], [0]], ART_REV: [[0], [0]],
                   ART_FK: [[], []]},
            hops={FK: [2, 0], REV: [0, 1], ART_REV: [0, 1], ART_FK: [0, 0]})
    row = row_for(fx)
    exactness = row['exactness']
    assert exactness['piece'] == 'interior' and exactness['boundary_neurons'] == 0
    assert exactness['min_abs_gated_preactivation'] == 0.0 and exactness['degree_t_claim']
    assert row['coverage']['total_walks'] == 7
    assert sorted(w['relevance'] for w in row['walks'])[:2] == [0.0, 0.0]  # never pruned
    oracle = GatedOracle(fx)
    masks = oracle.masks_at_one()
    flipped = copy.deepcopy(masks)
    flipped[('mp', 1, 'transactions')][1] = 1.0
    assert oracle.walk_scores(masks=masks) == oracle.walk_scores(masks=flipped)
    # A nonzero captured value of either sign where the float64 piece is
    # exactly 0 disagrees (contract 6.3).
    for value in (0.5, -0.5):
        computation = copy.deepcopy(computation_of(fx))
        computation.pre_activation[1]['transactions'][1, 0] = value
        exactness = run_gi(computation)[1]['exactness']
        assert exactness['mask_disagreements'] == 1 and exactness['boundary_neurons'] == 0
        assert exactness['reasons'] == ['mask_disagreement']


def _relevant_channel(computation, layer, positive=True):
    for table, occurrence in computation.relevant_nodes(layer):
        if layer == computation.T or table != computation.seed[0]:
            values = computation.pre_activation[layer][table][occurrence]
            for channel in range(values.numel()):
                if (values[channel] > 0) == positive and abs(float(values[channel])) > 1e-2:
                    return table, occurrence, channel
    raise AssertionError('no relevant channel')


def _row_from(fx, computation, enumeration, summary):
    """A row built from a (possibly tampered) computation with P2's builder."""
    from kumo_core.explain.walk_graph import architecture_record, operations_record
    selection = Selection('regression', 0, 1, None)
    output = {'task_type': 'regression', 'output_index': 0,
              'head_output': float(computation.head_output[0]), 'binary': None}
    return build_row(computation, enumeration, summary, config=CONFIG, selection=selection,
                     output=output,
                     qualification=list(STRUCTURAL_CODES) + list(gi.QUALIFY_CODES),
                     architecture=architecture_record(fx.module),
                     operations=operations_record(fx.module), sample=fx.sample(),
                     origin='original_forward_attribution', prediction_id=ID,
                     sample_sha256=SAMPLE_ID, numerics=computation.numerics)


def test_tampered_captures_withhold_the_claim_and_keep_the_identity():
    fx = fixture('relhm')
    computation = computation_of(fx)
    truth = scores_of(run_gi(computation)[0])
    # A flipped captured sign: the piece disagrees with its own float64 values.
    flipped = copy.deepcopy(computation)
    table, occurrence, channel = _relevant_channel(flipped, 1)
    flipped.pre_activation[1][table][occurrence, channel] *= -1
    enumeration, summary = run_gi(flipped)
    assert summary['exactness']['mask_disagreements'] >= 1
    assert 'mask_disagreement' in summary['exactness']['reasons']
    row = _row_from(fx, flipped, enumeration, summary)
    assert row['status'] == 'available' and not row['exactness']['degree_t_claim']
    assert abs(row['ledger']['accounting_error']) <= tolerance(1.0)
    validate_v2_row(row, fx.sample())
    assert scores_of(enumeration) != truth
    assert float(computation.pre_activation[1][table][occurrence, channel]) > 0  # untouched
    # An exact captured zero with nonzero sources is a boundary.
    zero = copy.deepcopy(computation)
    zero.pre_activation[1][table][occurrence, channel] = 0.0
    _, summary = run_gi(zero)
    assert summary['exactness']['boundary_neurons'] == 1
    assert summary['exactness']['reasons'][:1] == ['boundary_piece']
    # A head output 1e-3 away from the piece exceeds the `highest` bound only.
    offset = copy.deepcopy(computation)
    offset.head_output[0] += 1e-3
    enumeration, summary = run_gi(offset)
    assert summary['exactness']['reasons'] == ['precision_residual_exceeded']
    row = _row_from(fx, offset, enumeration, summary)
    assert abs(row['ledger']['precision_residual'] - 1e-3) < 1e-12
    validate_v2_row(row, fx.sample())
    medium = copy.deepcopy(offset)
    medium.numerics = {**offset.numerics, 'precision_mode': 'medium'}
    assert run_gi(medium)[1]['exactness']['degree_t_claim']
    # Routes that no longer match the retained edges are a capture defect.
    broken = copy.deepcopy(computation)
    relation = next(op for op in broken.relations[2] if op.relation == FK)
    relation.edge_index = relation.edge_index[:, :-1]
    with pytest.raises(ValueError, match='retained edges'):
        gi.make_rule(broken, select(), CONFIG)
    # So is a mean denominator that is not the frozen retained in-degree.
    broken = copy.deepcopy(computation)
    route = broken.incoming(2, broken.seed).routes[1]
    object.__setattr__(route, 'denominator', route.denominator + 1)  # a frozen record
    with pytest.raises(ValueError, match='frozen at the retained in-degree'):
        gi.make_rule(broken, select(), CONFIG)
    # And a piece whose output is not the masked pre-activation.
    broken = copy.deepcopy(computation)
    piece = broken.forward_piece()
    hidden = copy.deepcopy(piece.hidden)
    hidden[1][table][occurrence, channel] += 1.0
    broken.forward_piece = lambda: dataclasses.replace(piece, hidden=hidden)
    with pytest.raises(ValueError, match='masked pre-activation'):
        gi.make_rule(broken, select(), CONFIG)


def _scale_captured(computation, factor):
    """Every captured pre-activation times `factor` (in place on a copy)."""
    for t in range(1, computation.T + 1):
        for table in computation.pre_activation[t]:
            computation.pre_activation[t][table] *= factor
    for op in computation.preprocessing + computation.head:
        for table in op.pre_activation:
            op.pre_activation[table] *= factor


def test_margins_come_from_the_float64_piece_not_from_captured_values():
    # Review m1: scaling every captured pre-activation by 3 keeps every mask,
    # hence the piece, the walks, the counts and the claim, but would triple a
    # margin read from the captured values instead of `|a~|` of the piece.
    computation = computation_of(fixture('t3_trimmed'))  # preprocessing, MP and head neurons
    enumeration, reference = run_gi(computation)
    scaled = copy.deepcopy(computation)
    _scale_captured(scaled, 3.0)
    scaled_enumeration, summary = run_gi(scaled)
    assert summary['exactness'] == reference['exactness']
    assert summary['diagnostics'] == reference['diagnostics']
    assert scores_of(scaled_enumeration) == scores_of(enumeration)
    margins = reference['diagnostics']['gated_neurons']
    assert all(m is not None and m >= 1e-2 for kind in ('preprocessing', 'message_passing', 'head')
               for m in margins[kind]['min_abs_preactivation_by_layer'])


def _positive_preprocessing_neuron(computation):
    for table, occurrence in computation.relevant_nodes(0):
        values = computation.preprocessing[0].pre_activation[table][occurrence]
        for channel in range(values.numel()):
            if float(values[channel]) > 1e-2:
                return table, occurrence, channel
    raise AssertionError('no positive preprocessing neuron')


def test_preprocessing_neurons_are_counted_and_can_withhold_the_claim():
    # Review m1: preprocessing neurons of walk sources are relevant gated
    # neurons; a captured sign that the float64 piece contradicts withholds
    # the claim, and an exact captured zero with a nonzero input is a boundary.
    computation = computation_of(fixture('t3_trimmed'))
    summary = run_gi(computation)[1]
    preprocessing = summary['diagnostics']['gated_neurons']['preprocessing']
    assert preprocessing['relevant'] == len(computation.relevant_nodes(0)) * computation.channels
    assert summary['exactness']['degree_t_claim']
    table, occurrence, channel = _positive_preprocessing_neuron(computation)
    flipped = copy.deepcopy(computation)
    flipped.preprocessing[0].pre_activation[table][occurrence, channel] *= -1
    summary = run_gi(flipped)[1]
    assert summary['diagnostics']['gated_neurons']['preprocessing']['mask_disagreements'] == 1
    assert 'mask_disagreement' in summary['exactness']['reasons']
    assert not summary['exactness']['degree_t_claim']
    zero = copy.deepcopy(computation)
    zero.preprocessing[0].pre_activation[table][occurrence, channel] = 0.0
    summary = run_gi(zero)[1]
    assert summary['diagnostics']['gated_neurons']['preprocessing']['boundary'] == 1
    assert summary['exactness']['reasons'][0] == 'boundary_piece'


def test_readout_exact_zeros_follow_the_input_vector_rule():
    # Review m1: a seed whose last message-passing output is all zero (layer-2
    # customer root weight -10: a2(c) = -35 + 5.5 < 0) feeds an all-zero input
    # to the readout, whose hidden pre-activation is then exactly 0 for every
    # gate value near 1: interior, with a claim on all-zero walks.
    dead = b1(head_layers=2, values={'model.root_skips.1.lin_dict.customer.weight': [[-10.]]})
    row = row_for(dead)
    head = row['diagnostics'][DEGREE_T_METHOD]['gated_neurons']['head']
    assert head == {'relevant': 1, 'boundary': 0, 'mask_disagreements': 0,
                    'min_abs_preactivation_by_layer': [0.0]}
    assert row['exactness']['piece'] == 'interior' and row['exactness']['degree_t_claim']
    assert row['score'] == 0.0 and all(w['relevance'] == 0.0 for w in row['walks'])
    # A zero readout weight with a nonzero input also gives an exact zero; the
    # contract's rule is conservative and flags it as a boundary.
    zero_weight = b1(head_layers=2, values={'head.mlp.lins.0.weight': [[0.]]})
    row = row_for(zero_weight)
    assert row['diagnostics'][DEGREE_T_METHOD]['gated_neurons']['head']['boundary'] == 1
    assert row['exactness']['reasons'] == ['boundary_piece']


def _bf16_medium():
    """Whether `medium` runs float32 matmuls with bf16 internals on this host."""
    if mkldnn_matmul_fp32_precision() is None:
        return False
    with float32_matmul_precision('medium'):
        return mkldnn_matmul_fp32_precision() == 'bf16'


def _piece_margin(computation):
    """`min |a~|` of the float64 piece over relevant message-passing neurons."""
    piece = computation.forward_piece()
    return min(float(piece.pre_activation[t][table][occurrence].abs().min())
               for t in range(1, computation.T + 1)
               for table, occurrence in computation.relevant_nodes(t))


@pytest.mark.skipif(not _bf16_medium(),
                    reason='`medium` has no bf16 matmul path on this host, so captured float32 '
                    'values match the float64 piece up to rounding and the regime cannot occur')
def test_medium_precision_residuals_and_withheld_claims():
    """Rows under bf16 internals (12 rows per 32-channel linear).

    Every row closes its ledger with `precision_residual = score - f_piece`,
    reports the margin of the float64 piece (not of the captured values),
    withholds the claim whenever a mask disagrees, and, when claimed, equals
    the float64 capture bitwise. Which rows disagree depends on the host's
    kernels: on the development host (AMX, torch 2.13) 10 of the first 24 do.
    Elsewhere the test scans up to 200 rows for one and skips only if none
    occurs, saying so; the tampered-capture tests cover withheld claims
    deterministically on every host.
    """
    changed = disagreeing = claimed = 0
    for seed in range(200):
        if seed >= 24 and disagreeing:
            break
        row, computation = explained(precision_relhm(seed), mode='medium')
        highest = row_for(precision_relhm(seed), mode='highest')
        assert row['status'] == 'available', row.get('reason')
        reference = row_for(as_float64(precision_relhm(seed)))
        walks = {walk_key_of(w): w['relevance'] for w in row['walks']}
        exact = {walk_key_of(w): w['relevance'] for w in reference['walks']}
        diagnostics = row['diagnostics'][DEGREE_T_METHOD]
        ledger = row['ledger']
        assert ledger['precision_residual'] == row['score'] - diagnostics['f_piece']
        mass = math.fsum([abs(v) for v in walks.values()] + [abs(ledger['precision_residual'])])
        assert abs(ledger['accounting_error']) <= tolerance(mass)
        assert diagnostics['tolerance_mode'] == 'medium'
        assert row['exactness']['min_abs_gated_preactivation'] == _piece_margin(computation)
        changed += row['score'] != highest['score']
        if row['exactness']['mask_disagreements']:
            disagreeing += 1
            assert not row['exactness']['degree_t_claim']
            assert 'mask_disagreement' in row['exactness']['reasons']
            assert walks != exact  # not the float64 function's coefficients
        elif row['exactness']['degree_t_claim']:
            claimed += 1
            assert walks == exact  # bitwise: captured intermediates never enter scores
            if highest['exactness']['degree_t_claim']:
                assert walks == {walk_key_of(w): w['relevance'] for w in highest['walks']}
    if not changed:
        pytest.skip('`medium` left every probed float32 score unchanged on this host')
    # Under bf16 internals the prefix masses must still come from the float64
    # piece: the identity closes on an exhausted budget.
    budget = WalkAttributionConfig(**{**CONFIG.to_dict(), 'max_walks': 25, 'top_k': 5})
    row = row_for(precision_relhm(0), budget, mode='medium')
    assert row['status'] == 'incomplete', row.get('reason')
    assert row['ledger']['omitted_search_relevance'] != 0.0
    if not disagreeing:
        pytest.skip('no mask disagreement in 200 bf16 rows on this host: the per-row '
                    'invariants held, and the tampered-capture tests cover withheld claims')


def walk_key_of(walk):
    from kumo_core.explain.walk_schema import row_step_key
    return tuple(row_step_key(step) for step in walk['steps'])


def test_negative_sign_ledger_is_consistent():
    # Float32 module: the emitted score and the float64 piece differ slightly.
    rows = {}
    for selected in (True, False):
        fx = make_binary(relhm(seed_of('relhm'), dtype=torch.float32))
        config = WalkAttributionConfig(**{**CONFIG.to_dict(), 'selected_class': selected})
        rows[selected] = row_for(fx, config)
    positive, negative = rows[True], rows[False]
    z = negative['output']['head_output']
    diagnostics = negative['diagnostics'][DEGREE_T_METHOD]
    assert negative['output']['binary']['logit_sign'] == -1 and negative['score'] == -z
    assert diagnostics['f_piece'] == -diagnostics['z_piece']
    ledger = negative['ledger']
    assert ledger['precision_residual'] == negative['score'] - diagnostics['f_piece'] != 0.0
    walks = [w['relevance'] for w in negative['walks']]
    assert abs(ledger['accounting_error']) <= tolerance(math.fsum(abs(w) for w in walks))
    assert abs(math.fsum([negative['score'], -ledger['precision_residual']] + [-w for w in walks])
               - ledger['accounting_error']) <= 1e-15
    assert {walk_key_of(w): -w['relevance'] for w in positive['walks']} == {
        walk_key_of(w): w['relevance'] for w in negative['walks']}  # exact negation
    assert ledger['precision_residual'] == -positive['ledger']['precision_residual']
    # Negative control: the unsigned z_piece would leave an error of about 2|z|.
    unsigned = math.fsum([negative['score'], -(negative['score'] - diagnostics['z_piece'])]
                         + [-w for w in walks])
    assert abs(abs(unsigned) - 2 * abs(z)) < 1e-4 and abs(unsigned) > 0.1


# Rows ########################################################################

def test_regression_rows_explain_the_encoded_output():
    row = row_for(fixture('t3_trimmed'))
    assert row['status'] == 'available' and row['coordinate'] == 'encoded_model_output'
    assert row['output']['binary'] is None and row['exactness']['degree_t_claim']
    assert row['method_settings'] == gi.SETTINGS
    assert row['exactness']['qualification'] == list(gi.CLAIM_CODES)
    margins = row['diagnostics'][DEGREE_T_METHOD]['gated_neurons']
    assert len(margins['preprocessing']['min_abs_preactivation_by_layer']) == 1
    assert margins['head']['relevant'] == 3 and margins['message_passing']['relevant'] > 0


def test_rows_ignore_edge_type_insertion_order():
    fx = fixture('relhm')
    reordered = Fixture(fx.module, fx.x, dict(reversed(list(fx.edges.items()))), fx.nodes,
                        fx.hops)
    first, second = row_for(fx), row_for(reordered)
    keep = ('walks', 'sources', 'bags', 'ledger', 'exactness', 'coverage', 'diagnostics')
    assert {k: first[k] for k in keep} == {k: second[k] for k in keep}
    assert row_for(fixture('relhm')) == first  # deterministic


def _valid_row():
    row = row_for(fixture('t3_trimmed'))
    assert row['status'] == 'available' and row['exactness']['degree_t_claim']
    return row


def _claim(row):
    row['exactness'].update(decomposition='exact_degree_t', degree_t_claim=True, reasons=[])


MUTATIONS = {
    'claim_with_boundary': lambda r: r['exactness'].update(boundary_neurons=1,
                                                           piece='boundary'),
    'claim_with_disagreement': lambda r: r['exactness'].update(mask_disagreements=1,
                                                               piece='boundary'),
    'claim_with_incomplete_search': lambda r: r['coverage'].update(enumeration_complete=False),
    'claim_with_excessive_residual': lambda r: r['ledger'].update(precision_residual=1.0),
    'reasons_out_of_order': lambda r: r['exactness'].update(
        degree_t_claim=False, decomposition='piecewise_gnn_gi',
        reasons=['mask_disagreement', 'boundary_piece'], boundary_neurons=1,
        mask_disagreements=1, piece='boundary'),
    'duplicate_reason': lambda r: r['exactness'].update(
        degree_t_claim=False, decomposition='piecewise_gnn_gi',
        reasons=['boundary_piece', 'boundary_piece'], boundary_neurons=1, piece='boundary'),
    'bias_relevance': lambda r: r['ledger']['bias_relevance_by_kind'].update(head=0.0)
    or r['ledger'].update(bias_relevance=1e-3),
    'stabilizer_residual': lambda r: r['ledger'].update(stabilizer_residual=1e-3),
    'gamma_setting': lambda r: r['method_settings'].update(gamma=0.0),
    'unknown_piece': lambda r: r['exactness'].update(piece='edge'),
    'margin_without_gated_neurons': lambda r: r['exactness'].update(relevant_gated_neurons=0),
    'missing_claim_code': lambda r: r['exactness']['qualification'].remove('bias_free_gnn'),
    'duplicate_code': lambda r: r['exactness']['qualification'].append('interior_piece'),
    'row_code_without_fact': lambda r: r['exactness']['qualification'].remove(
        'tolerances_met'),
    'biased_architecture': lambda r: r['architecture'].update(gnn_bias=True),
    'bag_multiplicities': lambda r: r['bags'][0]['variables'][0].update(multiplicity=9),
    'bags_do_not_sum_to_f_piece': lambda r: r['bags'][0].update(
        relevance=r['bags'][0]['relevance'] + 1.0),
    'diagnostic_f_piece': lambda r: r['diagnostics'][DEGREE_T_METHOD].update(f_piece=1.0),
    'diagnostic_counts': lambda r: r['diagnostics'][DEGREE_T_METHOD]['gated_neurons'][
        'head'].update(relevant=0),
    'diagnostic_margin': lambda r: r['diagnostics'][DEGREE_T_METHOD]['gated_neurons'][
        'message_passing']['min_abs_preactivation_by_layer'].__setitem__(0, 1e-9),
    'diagnostic_tolerance': lambda r: r['diagnostics'][DEGREE_T_METHOD].update(
        claim_tolerance=1.0),
    'missing_diagnostics': lambda r: r.update(diagnostics=None),
}


@pytest.mark.parametrize('name', sorted(MUTATIONS))
def test_validator_rejects_mutation(name):
    row = _valid_row()
    validate_v2_row(row, fixture('t3_trimmed').sample())
    broken = copy.deepcopy(row)
    MUTATIONS[name](broken)
    with pytest.raises(WalkSchemaError):
        validate_v2_row(broken)


def test_validate_row_alone_checks_the_method_rules():
    row = _valid_row()
    gi.validate_row(row)
    broken = copy.deepcopy(row)
    broken['exactness']['reasons'] = ['precision_residual_exceeded']
    with pytest.raises(WalkSchemaError, match='claim tolerance'):
        gi.validate_row(broken)


# Independence of the oracle ####################################################

ORACLES = Path(__file__).with_name('walk_degree_t_oracles.py')


def test_the_oracle_imports_no_walk_module():
    imported = set()
    for node in ast.walk(ast.parse(ORACLES.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add('.' * node.level + (node.module or ''))
    assert imported and not [name for name in imported
                             if 'walk_' in name or name.startswith(('kumo_core.explain',
                                                                    'tests', '.'))]
    # Transitively too: importing the oracle loads no walk module.
    code = ('import sys; sys.path.insert(0, sys.argv[1]); import walk_degree_t_oracles; '
            'print(sorted(m for m in sys.modules if m.startswith("kumo_core.explain.walk")))')
    result = subprocess.run([sys.executable, '-c', code, str(ORACLES.parent)],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == '[]'


# Cost on the Rel-HM shape #####################################################

def test_relhm_shape_structure():
    # The row size behind the cost estimate (timing is measured in the
    # handoff, not asserted, so shared CI hosts cannot flake this test).
    row = row_for(relhm(0, channels=32))
    assert row['status'] == 'available'
    assert row['coverage']['total_walks'] == 33 and row['coverage']['expanded_routes'] == 42
    assert row['exactness']['relevant_gated_neurons'] == 320
    assert len(row['bags']) == 33 and all(b['walk_count'] == 1 for b in row['bags'])
    assert sum(any(v['multiplicity'] == 2 for v in b['variables']) for b in row['bags']) == 1
