# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`hetero_gnn_lrp_weight_gamma_v1`: hand fixtures, oracles and accounting.

The weight rule is checked against hand fractions (paper fixtures B2 and B3,
bias signs, near-zero denominators, fixture B1 with biases and the
head sign-inversion fixture H1), against the legacy contribution rule where
the two must coincide (nonnegative inputs), against Engine's v1 capture, and
against an independent autograd forward-rewrite oracle
(`tests/walk_lrp_oracles.py`, paper Section III-B1) on signed and T = 3
graphs. Ledger closure is checked everywhere but never taken as evidence of
correctness on its own.
"""
import ast
import copy
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from kumoapi.encoder import Index
from kumoapi.task import TaskType
from kumoapi.typing import ColStatType
from torch_geometric.nn.aggr import MaxAggregation

from kumo_core.encoders.target.base import TargetEncoder
from kumo_core.explain import walk_lrp_weight_gamma as weight_gamma
from kumo_core.explain.walk_attribution import (
    LEGACY_METHOD,
    WEIGHT_GAMMA_METHOD,
    UnsupportedWalkAttribution,
    WalkAttributionConfig,
    WalkCapture,
    inspect_walk_capability,
    method_module,
)
from kumo_core.explain.walk_binary import (
    Selection,
    output_block,
    resolve_selection,
)
from kumo_core.explain.walk_enumerate import enumerate_walks
from kumo_core.explain.walk_graph import STRUCTURAL_CODES, GraphCapture
from kumo_core.explain.walk_lrp_reference import (
    ContributionGammaRule,
    epsilon_split,
)
from kumo_core.explain.walk_lrp_reference import qualify as legacy_qualify
from kumo_core.explain.walk_schema import (
    WalkSchemaError,
    build_row,
    explain_capture,
    row_numerics,
    validate_v2_row,
    walk_key,
)
from kumo_core.modeling.calibration import PlatScaling
from kumo_core.modeling.gnn.hetero import (
    HeteroGNN,
    HeteroLinear,
    PrimaryKeyToForeignKeyConv,
)
from kumo_core.modeling.heads.node_prediction import NodePredHead
from kumo_core.train.loss import BinaryCrossEntropyLoss
from tests import walk_lrp_oracles as oracle
from tests.test_walk_graph import (
    ART_FK,
    ART_REV,
    FK,
    REV,
    TABLES,
    b1_batch,
    b1_module,
    capture,
)

F64 = torch.float64
WEIGHT = WEIGHT_GAMMA_METHOD
ID, SAMPLE_ID = 'a' * 64, 'b' * 64
NUMERICS = {'precision_mode': 'highest', 'mkldnn_matmul_fp32_precision': 'ieee',
            'num_threads': 2, 'cpu_capability': 'AVX512', 'torch_version': '2.13.0+cpu'}
PHI = 98 / 123  # the B1-bias head factor at gamma 0.25 (P3 brainstorm 5.1)


def config(**options):
    return WalkAttributionConfig(method=WEIGHT, **options)


def tensor(values):
    return torch.tensor(values, dtype=F64)


def no_hooks(module):
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in module.modules())


# Builders ######################################################################

def b1(dtype=F64, *, head_bias=None, gnn_bias=False, **options):
    """B1 (all weights 1, mean, learned roots); `head_bias` sets the head bias
    (B1-bias: GNN biases 0.5 and head bias -0.25 give z = 12.25; H1: a
    bias-free GNN and head bias -9.5 give z = -0.5)."""
    values = None if head_bias is None else {'head.mlp.lins.0.bias': [head_bias]}
    return b1_module(dtype, gnn_bias=gnn_bias, head_bias=head_bias is not None,
                     values=values, **options), (*b1_batch(dtype), )


def forward(module, batch):
    x, edges, nodes, counts = batch
    nodes = None if nodes is None else {k: list(v) for k, v in nodes.items()}
    counts = None if counts is None else {k: list(v) for k, v in counts.items()}
    return module.head(module.model(dict(x), dict(edges), nodes,
                                    counts)[module.entity_table_name][:1])


def relhm_module(seed, *, T=2, channels=3, pre=0, head_layers=1, aggr='mean',
                 learn_skip=True, gnn_bias=True, head_bias=True, act='relu'):
    """A random float64 Rel-HM-shaped scorer (customer seed, transactions,
    articles) with signed weights."""
    generator = torch.Generator().manual_seed(seed)
    module = cast(Any, torch.nn.Module())  # a scoring-module stand-in
    module.entity_table_name = 'customer'
    module.train_table_name = 'task'
    module.model = HeteroGNN(channels, T, metadata=(TABLES, [FK, REV, ART_REV, ART_FK]),
                             num_pre_mp_layers=pre, aggr=[aggr], norm=None, bias=gnn_bias,
                             learn_skip=learn_skip, act=act)
    module.head = NodePredHead(channels, 1, num_layers=head_layers, norm=None,
                               param_keys=[], bias=head_bias, act=act)
    module = module.to(F64).eval()
    with torch.no_grad():
        for name, param in module.named_parameters():
            scale = 0.3 if name.endswith('bias') else 0.7
            param.copy_(scale * torch.randn(param.shape, generator=generator, dtype=F64))
    return module


def relhm_batch(seed, *, T=2, channels=3, n=3, k=2, m=2, signed=False, zero_rows=(),
                zero_channels=(), duplicate_fk=False):
    """Sampled Rel-HM neighborhoods with Engine trimming.

    T = 2: the seed customer, `n` hop-1 transactions (FK into the seed, REV
    back) and `k` hop-2 articles. T = 3 adds `m` hop-3 transactions of those
    articles, so walks such as t -> c -> t -> c repeat edges. Nonnegative
    features are clamped normal draws (so some inputs are exactly zero);
    `duplicate_fk` adds a parallel FK edge from transaction 0 to the seed.
    """
    generator = torch.Generator().manual_seed(1000 + seed)
    far = m if T == 3 else 0

    def features(rows):
        x = torch.randn(rows, channels, generator=generator, dtype=F64)
        return x if signed else x.clamp(min=0)

    x = {'customer': features(1), 'transactions': features(n + far), 'article': features(k)}
    for table, row in zero_rows:
        x[table][row] = 0.0
    for table, column in zero_channels:
        x[table][:, column] = 0.0
    fk = [list(range(n)) + ([0] if duplicate_fk else []), [0] * (n + duplicate_fk)]
    edges = {FK: torch.tensor(fk), REV: torch.tensor([[0] * n, list(range(n))]),
             ART_REV: torch.tensor([[i % k for i in range(n)], list(range(n))]),
             ART_FK: (torch.tensor([list(range(n, n + far)), [j % k for j in range(far)]])
                      if far else torch.zeros(2, 0, dtype=torch.long))}
    if T == 3:
        nodes = {'customer': [1, 0, 0, 0], 'transactions': [0, n, 0, far],
                 'article': [0, 0, k, 0]}
        counts = {FK: [n + duplicate_fk, 0, 0], REV: [0, n, 0], ART_REV: [0, n, 0],
                  ART_FK: [0, 0, far]}
    else:
        nodes = {'customer': [1, 0, 0], 'transactions': [0, n, 0], 'article': [0, 0, k]}
        counts = {FK: [n + duplicate_fk, 0], REV: [0, n], ART_REV: [0, n], ART_FK: [0, 0]}
    return x, edges, nodes, counts


def tiny(x, head_weight, *, head_bias=None, act=None, root=None):
    """T = 1 and no edges: `H[1, seed] = act(root(x))` and a one-linear head
    `(w, b)`, so the walk vector is the head step's split (B2, B3, biases)."""
    relation = ('event', 'user_id', 'user')
    module = cast(Any, torch.nn.Module())  # a scoring-module stand-in
    module.entity_table_name = 'user'
    module.train_table_name = 'task'
    channels = len(x)
    module.model = HeteroGNN(channels, 1, metadata=(['user', 'event'], [relation]),
                             aggr=['mean'], norm=None, bias=False,
                             learn_skip=root is not None, act=act)
    module.head = NodePredHead(channels, 1, num_layers=1, norm=None, param_keys=[],
                               bias=head_bias is not None)
    module = module.to(F64).eval()
    with torch.no_grad():
        module.head.mlp.lins[0].weight.copy_(tensor([head_weight]))
        if head_bias is not None:
            module.head.mlp.lins[0].bias.fill_(head_bias)
        if root is not None:
            for lin in module.model.root_skips[0].lin_dict.values():
                lin.weight.copy_(tensor(root))
    batch = ({'user': tensor([x]), 'event': torch.zeros(0, channels, dtype=F64)},
             {relation: torch.zeros(2, 0, dtype=torch.long)}, None, None)
    return module, batch


def as_binary(module, counts=((True, 30), (False, 10))):
    """Attach a BCE loss, a fitted two-label target encoder and an unfitted
    Platt calibrator, so `selected_class` chooses `logit_sign`."""
    module.task_type = TaskType.BINARY_CLASSIFICATION
    module.loss = BinaryCrossEntropyLoss(TaskType.BINARY_CLASSIFICATION)
    index = Index()
    index._stats = {ColStatType.CATEGORY_COUNTS: [list(c) for c in counts]}
    module.target_encoder = TargetEncoder.from_encoder(TaskType.BINARY_CLASSIFICATION, index)
    module.calibrator = PlatScaling(TaskType.BINARY_CLASSIFICATION).to(F64)
    return module.eval()


def sample_of(batch):
    return {'nodes': {table: {'pkeys': [100 * i + j for j in range(x.size(0))],
                              'node_ids': [1000 * i + j for j in range(x.size(0))]}
                      for i, (table, x) in enumerate(sorted(batch[0].items()))}}


def rows_for(module, batch, configs):
    """One captured forward, then one row per config through the shared row
    builder (which validates every available or incomplete row)."""
    binary = getattr(module, 'task_type', None) == TaskType.BINARY_CLASSIFICATION
    with torch.no_grad(), GraphCapture(module, configs[0]) as cap:
        logits = forward(module, batch)
    rows = []
    for cfg in configs:
        selection = (resolve_selection(module, cfg) if binary else
                     Selection('regression', cfg.output_index, 1, None))
        rows.append(explain_capture(
            cap, config=cfg, selection=selection,
            output=output_block(selection, module, logits[0]), sample=sample_of(batch),
            origin='original_forward_attribution', prediction_id=ID,
            sample_sha256=SAMPLE_ID, numerics=row_numerics(NUMERICS, 'float64')))
    return rows


def computation_of(module, batch):
    cap, _ = capture(module, lambda: forward(module, batch))
    return cap.computation()


def enumerate_rule(module, batch, cfg, *, sign=1, rule=None):
    computation = computation_of(module, batch)
    selection = Selection('regression', cfg.output_index, sign, None)
    make = rule or weight_gamma.make_rule
    return computation, selection, enumerate_walks(computation, make(computation, selection,
                                                                     cfg),
                                                   max_walks=cfg.max_walks)


def row_key(walk):
    """The contract walk key of a serialized walk (layer 1 first)."""
    return tuple((s['mp_layer'], 'relation', *s['relation'], s['edge_occurrence'])
                 if s['kind'] == 'relation' else
                 (s['mp_layer'], 'self_update', s['table'], '', s['table'], s['occurrence'])
                 for s in walk['steps'])


def walks_of(row):
    return {row_key(w): w['relevance'] for w in row['walks']}


def mass(row):
    """`A`, the absolute ledger mass of a row (contract 4.10)."""
    ledger = row['ledger']
    return math.fsum([abs(w['relevance']) for w in row['walks']]
                     + [ledger['omitted_display_abs_relevance'],
                        ledger['omitted_search_abs_prefix_mass'],
                        abs(ledger['precision_residual'])]
                     + [abs(v) for v in ledger['bias_relevance_by_kind'].values()]
                     + [abs(v) for v in ledger['stabilizer_residual_by_kind'].values()])


def keyed(enumeration):
    return {walk_key(w.steps): w.relevance for w in enumeration.walks}


SELF_C1 = (1, 'self_update', 'customer', '', 'customer', 0)
SELF_C2 = (2, 'self_update', 'customer', '', 'customer', 0)


def rel(t, relation, edge):
    return (t, 'relation', *relation, edge)


#: The eight B1 walks and their scores at gamma 0 without biases (sum 9).
B1_SCORES = {
    (SELF_C1, SELF_C2): 1.0,
    (rel(1, FK, 0), SELF_C2): 1.0,
    (rel(1, FK, 1), SELF_C2): 1.5,
    ((1, 'self_update', 'transactions', '', 'transactions', 0), rel(2, FK, 0)): 1.0,
    (rel(1, REV, 0), rel(2, FK, 0)): 0.5,
    (rel(1, ART_REV, 0), rel(2, FK, 0)): 2.0,
    ((1, 'self_update', 'transactions', '', 'transactions', 1), rel(2, FK, 1)): 1.5,
    (rel(1, REV, 1), rel(2, FK, 1)): 0.5,
}


# Registration and qualification ###############################################

def test_registration_configuration_and_inspection():
    assert method_module(WEIGHT) is weight_gamma
    assert weight_gamma.METHOD_ID == 'hetero_gnn_lrp_weight_gamma_v1'
    cfg = config(gamma=0.25, feature_vectors=True, selected_class=True)
    assert cfg.to_dict()['method'] == WEIGHT and WalkAttributionConfig(**cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match='bags=True requires'):
        config(bags=True)
    module, _ = b1()
    report = inspect_walk_capability(module, config=config())
    assert report['supported'] and report['qualification'] == list(STRUCTURAL_CODES)
    assert no_hooks(module)


def _swap_class(instance, base):
    instance.__class__ = type(f'Other{base.__name__}', (base, ), {})


def _post_operator(module):
    module.model.post_convs[0].layer_dict['customer'][0] = torch.nn.GELU()


def _actual_aggregator(module):
    module.model.hetero_convs[0][FK].aggr = MaxAggregation()


#: Contract section 7 rejections that apply to modules (both LRP rules reject).
REJECTIONS = {
    'model_type': lambda m: _swap_class(m.model, HeteroGNN),
    'head_type': lambda m: _swap_class(m.head, NodePredHead),
    'training_mode': lambda m: m.train(),
    'gelu': lambda m: setattr(m.model, 'act', 'gelu'),
    'leaky_relu': lambda m: setattr(m.model, 'act', 'leaky_relu'),
    'elu_head': lambda m: setattr(m.head, 'act', 'elu'),
    'batch_norm': lambda m: setattr(m.model, 'norm', 'batch_norm'),
    'layer_norm_head': lambda m: setattr(m.head, 'norm', 'layer_norm'),
    'prediction_time_encodings': lambda m: setattr(m.head, 'prediction_time_encodings', True),
    'distribution_head': lambda m: setattr(m.head, 'param_keys', ['mu']),
    'max_aggregation': lambda m: setattr(m.model, 'aggr', ['max']),
    'multi_aggregation': lambda m: setattr(m.model, 'aggr', ['sum', 'mean']),
    'post_operator': _post_operator,
    'root_operator': lambda m: _swap_class(m.model.root_skips[0], HeteroLinear),
    'relation_operator': lambda m: _swap_class(m.model.hetero_convs[0][ART_REV],
                                               PrimaryKeyToForeignKeyConv),
    'actual_aggregator': _actual_aggregator,
}


@pytest.mark.parametrize('name', sorted(REJECTIONS))
def test_rejections_match_the_legacy_rule_and_precede_hooks(name):
    module, batch = b1(gnn_bias=True, head_bias=-0.25)
    REJECTIONS[name](module)
    with pytest.raises(UnsupportedWalkAttribution):
        legacy_qualify(module, WalkAttributionConfig())
    with pytest.raises(UnsupportedWalkAttribution):
        weight_gamma.qualify(module, config())
    with pytest.raises(UnsupportedWalkAttribution):
        with GraphCapture(module, config()):
            pass
    assert no_hooks(module)


def test_identity_activation_and_eval_mode_dropout_are_allowed():
    module, batch = b1(act=None)
    assert weight_gamma.qualify(module, config()) == list(STRUCTURAL_CODES)
    module, batch = b1(gnn_bias=True, head_bias=-0.25)
    for post in module.model.post_convs:
        for layers in post.layer_dict.values():
            layers[-1].p = 0.5
    (row, ) = rows_for(module, batch, [config(gamma=0.25)])
    assert row['status'] == 'available', row.get('reason')


# Local step by hand ###########################################################

def rule_for(gamma, rule_class=weight_gamma.WeightGammaRule):
    """A rule instance; its contribution only reads `gamma` (and `epsilon`)."""
    module, batch = tiny([1.0, 2.0], [1.0, 1.0])
    cfg = (config(gamma=gamma) if rule_class is weight_gamma.WeightGammaRule
           else WalkAttributionConfig(gamma=gamma))
    return rule_class(computation_of(module, batch), Selection('regression', 0, 1, None), cfg)


def split(rule, x, w, relevance, *, bias=None, epsilon=1e-300):
    """One output neuron through the rule's contribution and P2's epsilon split."""
    biases = [] if bias is None else [rule.modified_bias(tensor([bias]))]
    children, shares, residual = epsilon_split(
        [rule.modified_contribution(tensor(x), tensor([w]), 1)], biases,
        tensor([relevance]), epsilon)
    return children[0].tolist(), math.fsum(shares), residual


B2 = [((1.0, 2.0), (2.0, -0.5), (5 / 3, -2 / 3), (5 / 3, -2 / 3)),
      ((-1.0, 2.0), (2.0, 1.5), (-2.0, 3.0), (-8 / 7, 15 / 7))]


@pytest.mark.parametrize('x,w,weight,contribution', B2)
def test_b2_signed_case_by_hand(x, w, weight, contribution):
    """P0 fixture B2 at gamma 0.25: the weight rule enhances `x * w` with a
    positive weight even when `x < 0`; the contribution rule enhances
    positive products. Both give `z_j` at gamma 0."""
    z = x[0] * w[0] + x[1] * w[1]
    for rule_class, expected in ((weight_gamma.WeightGammaRule, weight),
                                 (ContributionGammaRule, contribution)):
        children, _, _ = split(rule_for(0.25, rule_class), x, w, z)
        assert children == pytest.approx(expected, abs=1e-12)
        children, _, _ = split(rule_for(0.0, rule_class), x, w, z)
        assert children == pytest.approx([x[0] * w[0], x[1] * w[1]], abs=1e-12)
    # Through Engine modules: identity root, identity activation, head w.
    module, batch = tiny(list(x), list(w))
    for method, expected in ((WEIGHT, weight), (LEGACY_METHOD, contribution)):
        (row, ) = rows_for(module, batch, [WalkAttributionConfig(
            method=method, gamma=0.25, epsilon=1e-300, feature_vectors=True)])
        assert row['status'] == 'available', row.get('reason')
        (walk, ) = row['walks']
        assert walk['input_relevance'] == pytest.approx(expected, abs=1e-12)
        assert row['ledger']['stabilizer_residual'] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize('bias,inputs,share', [
    (1.0, (1.0, 2.0), 1.0),
    (-1.0, (10 / 11, 20 / 11), -8 / 11),
])
def test_bias_signs_at_gamma_quarter_by_hand(bias, inputs, share):
    """`b + gamma * max(0, b)`: a positive bias is enhanced like a positive
    weight; a negative one absorbs negative relevance and inflates inputs."""
    for rule_class in (weight_gamma.WeightGammaRule, ContributionGammaRule):
        children, shares, _ = split(rule_for(0.25, rule_class), (1.0, 2.0), (1.0, 1.0),
                                    3.0 + bias, bias=bias)
        assert children == pytest.approx(list(inputs), abs=1e-12)
        assert shares == pytest.approx(share, abs=1e-12)
    module, batch = tiny([1.0, 2.0], [1.0, 1.0], head_bias=bias, act='relu')
    (row, ) = rows_for(module, batch, [config(gamma=0.25, epsilon=1e-300,
                                              feature_vectors=True)])
    (walk, ) = row['walks']
    assert row['score'] == 3.0 + bias
    assert walk['input_relevance'] == pytest.approx(list(inputs), abs=1e-12)
    assert row['ledger']['bias_relevance_by_kind'] == pytest.approx(
        {'preprocessing': 0.0, 'message_passing': 0.0, 'head': share}, abs=1e-12)


@pytest.mark.parametrize('w', [(1.0, -1.0), (1.0, -(1 - 2 ** -31)), (-1.0, 1 - 2 ** -31),
                               (1.0, -1 + 1e-3)])
def test_near_zero_denominators_and_epsilon_accounting(w):
    """Totals 0, 2^-31, -2^-31 and 1e-3: `|D| >= epsilon`, the stabilizer
    takes `epsilon / |D|` of the relevance with its sign, and the shares
    stay bounded by `|R| / epsilon`."""
    rule = rule_for(0.0)
    for relevance in (1.0, -1.0):
        children, _, residual = split(rule, (1.0, 1.0), w, relevance, epsilon=1e-9)
        total = w[0] + w[1]
        denominator = total + (1e-9 if total >= 0 else -1e-9)
        assert residual == pytest.approx(relevance * 1e-9 / abs(denominator), rel=1e-6)
        assert math.fsum(children) + residual == pytest.approx(relevance, abs=1e-6)
        assert all(abs(c) <= abs(relevance) / 1e-9 * 1.0000001 for c in children)


def test_exact_float64_epsilon_with_the_sign_of_the_total():
    rule = rule_for(0.0)
    # A total of exactly 0 takes +epsilon: the shares are R / epsilon with the
    # float64 epsilon (v1's float32-effective epsilon would give 1000000028).
    children, _, residual = split(rule, (1.0, 1.0), (1.0, -1.0), 1.0, epsilon=1e-9)
    assert children == [1.0 / 1e-9, -1.0 / 1e-9] and residual == 1.0
    children, _, _ = split(rule, (1.0, 1.0), (1.0, -1.0), -1.0, epsilon=1e-9)
    assert children == [-1.0 / 1e-9, 1.0 / 1e-9]
    # A tiny negative total takes -epsilon.
    children, _, _ = split(rule, (1.0, 1.0), (-1.0, 1 - 2 ** -31), 1.0, epsilon=1e-9)
    assert children[0] == -1.0 / (-(2 ** -31) - 1e-9)
    assert float(torch.tensor(1e-9, dtype=torch.float32)) != 1e-9


@pytest.mark.parametrize('x,w,gamma,weight,contribution', [
    ((1.0, 2.0), (2.0, -0.5), 0.25, (-8 / 3, 5 / 3), (-8 / 3, 5 / 3)),
    ((1.0, 2.0), (2.0, -0.5), 0.0, (-2.0, 1.0), (-2.0, 1.0)),
    ((-1.0, 2.0), (2.0, 1.5), 0.25, (2.0, -3.0), (5.0, -6.0)),
])
def test_b3_sign_folding_by_hand(x, w, gamma, weight, contribution):
    """Fixture B3: `s = -1` folds into the head row before the rule."""
    module, batch = tiny(list(x), list(w))
    cfg_w = config(gamma=gamma, epsilon=1e-300)
    cfg_c = WalkAttributionConfig(gamma=gamma, epsilon=1e-300)
    for cfg, rule, expected in ((cfg_w, None, weight),
                                (cfg_c, ContributionGammaRule, contribution)):
        vectors = {}
        for sign in (1, -1):
            _, _, enumeration = enumerate_rule(module, batch, cfg, sign=sign, rule=rule)
            (walk, ) = enumeration.walks
            vectors[sign] = walk.input_relevance.tolist()
        assert vectors[-1] == pytest.approx(list(expected), abs=1e-12)
        negated = [-v for v in vectors[1]]
        if gamma == 0:
            assert vectors[-1] == negated  # folding is negation at gamma 0
        elif rule is ContributionGammaRule or x[0] > 0:
            assert max(abs(a - b) for a, b in zip(vectors[-1], negated)) > 0.9


def test_reference_gate_signed_fixture_by_hand():
    """Sum aggregation, identity root, ReLU, events (-2, 4), entity 1, unit
    weights, head weight 3 (z = 9), gamma 1: the weight rule enhances the
    negative input through a positive weight (denominator 6, not 8)."""
    relation = ('events', 'entity_id', 'entities')
    module = cast(Any, torch.nn.Module())  # a scoring-module stand-in
    module.entity_table_name = 'entities'
    module.train_table_name = 'task'
    module.model = HeteroGNN(1, 1, metadata=(['events', 'entities'], [relation]), aggr='sum',
                             act='relu', norm=None, bias=False, learn_skip=False)
    module.head = NodePredHead(1, 1, num_layers=1, norm=None, param_keys=[], bias=False)
    module = module.to(F64).eval()
    with torch.no_grad():
        module.model.hetero_convs[0][relation].lin.weight.fill_(1.0)
        module.head.mlp.lins[0].weight.fill_(3.0)
    batch = ({'events': tensor([[-2.0], [4.0]]), 'entities': tensor([[1.0]])},
             {relation: torch.tensor([[0, 1], [0, 0]])}, None, None)
    expected = {WEIGHT: {('entities', 0): 3.0, ('events', 0): -6.0, ('events', 1): 12.0},
                LEGACY_METHOD: {('entities', 0): 2.25, ('events', 0): -2.25,
                                ('events', 1): 9.0}}
    for method, values in expected.items():
        (row, ) = rows_for(module, batch, [WalkAttributionConfig(method=method, gamma=1.0)])
        got = {(w['source_table'], w['source_occurrence']): w['relevance']
               for w in row['walks']}
        assert got == pytest.approx(values, abs=1e-7)
        assert math.fsum(got.values()) == pytest.approx(9.0, abs=1e-7)


# Engine B1 by hand ##############################################################

@pytest.mark.parametrize('gamma,sign,factor,mp_bias,head_bias', [
    (0.0, 1, Fraction(1), Fraction(7, 2), Fraction(-1, 4)),
    (0.25, 1, Fraction(245, 246), Fraction(7, 2) * Fraction(245, 246),
     Fraction(-1, 4) * Fraction(98, 123)),
    (0.0, -1, Fraction(-1), Fraction(-7, 2), Fraction(1, 4)),
    (0.25, -1, Fraction(-196, 195), Fraction(-7, 2) * Fraction(196, 195),
     Fraction(5, 16) * Fraction(196, 195)),
])
def test_b1_with_biases_by_hand(gamma, sign, factor, mp_bias, head_bias):
    """B1-bias (GNN biases 0.5, head bias -0.25, z = 12.25): every walk is a
    uniform factor times its B1 score; folding (`s = -1`) is not negation at
    gamma 0.25 (-196/195 versus -245/246)."""
    module, batch = b1(gnn_bias=True, head_bias=-0.25)
    _, _, enumeration = enumerate_rule(module, batch, config(gamma=gamma, top_k=100),
                                       sign=sign)
    walks = keyed(enumeration)
    assert set(walks) == set(B1_SCORES)
    for key, score in B1_SCORES.items():
        assert walks[key] == pytest.approx(float(factor) * score, abs=1e-7)
    assert enumeration.bias['message_passing'] == pytest.approx(float(mp_bias), abs=1e-7)
    assert enumeration.bias['head'] == pytest.approx(float(head_bias), abs=1e-7)
    assert enumeration.bias['preprocessing'] == 0.0
    total = (math.fsum(walks.values()) + math.fsum(enumeration.bias.values())
             + math.fsum(enumeration.stabilizer.values()))
    assert total == pytest.approx(sign * 12.25, abs=1e-12)


@pytest.mark.parametrize('gamma,sign,factor,head_bias,denominator,flipped', [
    (0.0, 1, Fraction(1), Fraction(-19, 2), -0.5, False),
    (0.25, 1, Fraction(-5, 14), Fraction(19, 7), 1.75, True),
    (0.5, 1, Fraction(-3, 16), Fraction(19, 16), 4.0, True),
    (0.25, -1, Fraction(-4, 23), Fraction(95, 46), 2.875, False),
])
def test_head_sign_inversion_fixture_h1(gamma, sign, factor, head_bias, denominator, flipped):
    """H1: a bias-free B1 GNN and head bias -9.5 give z = -0.5. With
    `gamma > 0` the folded, gamma-modified output total turns positive, so
    every walk changes sign (the graph seems to oppose churn) and the head
    bias absorbs positive relevance; rows report it in the diagnostics. The
    other class (`s = -1`, score +0.5) is not inverted."""
    module, batch = b1(head_bias=-9.5)
    module = as_binary(module)
    (row, ) = rows_for(module, batch, [config(gamma=gamma, top_k=100,
                                              selected_class=sign == 1)])
    assert row['status'] == 'available', row.get('reason')
    assert row['score'] == sign * -0.5
    walks = walks_of(row)
    for key, score in B1_SCORES.items():
        assert walks[key] == pytest.approx(float(factor) * score, abs=1e-7)
    assert row['ledger']['bias_relevance_by_kind']['head'] == pytest.approx(
        float(head_bias), abs=1e-7)
    diagnostics = row['diagnostics'][WEIGHT]
    assert diagnostics['head_output_denominator'] == pytest.approx(denominator, abs=1e-12)
    assert diagnostics['head_output_flipped'] is flipped
    assert diagnostics['denominator_sign_flips_by_kind'] == {
        'preprocessing': 0, 'message_passing': 0, 'head': int(flipped)}


def test_zero_head_denominator_is_bounded_by_epsilon():
    """H1 at gamma 1/18: the output total is exactly 0, so `D = +epsilon`, the
    head stabilizer takes the whole selected logit and shares reach
    `|R| / epsilon`; the ledger still closes within `tol(A)`."""
    module, batch = b1(head_bias=-9.5)
    (row, ) = rows_for(module, batch, [config(gamma=1 / 18, top_k=100)])
    assert row['status'] == 'available', row.get('reason')
    diagnostics = row['diagnostics'][WEIGHT]
    assert diagnostics['head_output_denominator'] == 0.0
    assert diagnostics['head_output_flipped'] is True
    assert diagnostics['min_denominator_margin'] == 0.0
    assert row['ledger']['stabilizer_residual_by_kind']['head'] == pytest.approx(-0.5,
                                                                                  abs=1e-12)
    assert max(abs(w['relevance']) for w in row['walks']) > 1e8
    assert abs(row['ledger']['accounting_error']) <= 1e-9 * max(1.0, mass(row))


# Budgets, display and the prefix masses ##########################################

@pytest.mark.parametrize('budget,expanded,walks,prefixes,displayed,search,bias', [
    (3, 3, 0, {('customer', 0): 45 / 8, ('transactions', 0): 85 / 16,
               ('transactions', 1): 55 / 16}, 0.0, 14.375, 1.0),
    (6, 6, 3, {('transactions', 0): 85 / 16, ('transactions', 1): 55 / 16}, 4.375, 8.75, 2.25),
    (8, 8, 5, {('transactions', 0): 85 / 16}, 6.875, 85 / 16, 3.1875),
    (11, 11, 8, {}, 11.25, 0.0, 4.125),
])
def test_budget_omissions_by_hand(budget, expanded, walks, prefixes, displayed, search, bias):
    """B1-bias at gamma 0.25 (values in units of phi = 98/123): exact signed
    prefix masses, budgeted walks bitwise equal to unbudgeted ones, and every
    row summing to `15.375 phi = 12.25`."""
    module, batch = b1(gnn_bias=True, head_bias=-0.25)
    (row, full) = rows_for(module, batch, [config(gamma=0.25, max_walks=budget, top_k=100),
                                           config(gamma=0.25, top_k=100)])
    complete = budget == 11
    assert row['status'] == ('available' if complete else 'incomplete'), row.get('reason')
    coverage, ledger = row['coverage'], row['ledger']
    assert coverage['expanded_routes'] == expanded and coverage['total_walks'] == walks
    assert coverage['status'] == ('complete' if complete else 'budget_exhausted')
    assert coverage['unexpanded_prefixes'] == len(prefixes)
    assert ledger['displayed_walk_relevance'] == pytest.approx(displayed * PHI, abs=1e-7)
    assert ledger['omitted_search_relevance'] == pytest.approx(search * PHI, abs=1e-7)
    assert ledger['omitted_search_abs_prefix_mass'] == pytest.approx(search * PHI, abs=1e-7)
    assert ledger['bias_relevance'] == pytest.approx(bias * PHI, abs=1e-7)
    assert ('enumeration_incomplete' in row['exactness']['reasons']) is not complete
    unbudgeted = walks_of(full)
    assert all(unbudgeted[key] == value for key, value in walks_of(row).items())
    # The same masses from the enumeration, by unexpanded destination.
    _, _, enumeration = enumerate_rule(module, batch, config(gamma=0.25, max_walks=budget))
    assert {p.destination: p.mass for p in enumeration.prefixes} == pytest.approx(
        {node: value * PHI for node, value in prefixes.items()}, abs=1e-7)


def test_prefix_masses_are_walks_plus_absorption_below_them():
    """A prefix mass is the relevance of every walk completing it plus the
    bias and stabilizer shares its subtree would absorb (I8); for LRP it is
    not the omitted walk relevance."""
    module, batch = b1(gnn_bias=True, head_bias=-0.25)
    cfg = config(gamma=0.25)
    _, _, full = enumerate_rule(module, batch, cfg)
    below = {('customer', 0): 0.9959350, ('transactions', 0): 0.7469512,
             ('transactions', 1): 0.7469512}
    for budget in (3, 6, 8):
        _, _, cut = enumerate_rule(module, batch, config(gamma=0.25, max_walks=budget))
        for prefix in cut.prefixes:
            through = [w.relevance for w in full.walks
                       if walk_key(w.steps)[prefix.layer:] == walk_key(prefix.steps)
                       and w.steps[prefix.layer - 1].destination == prefix.destination]
            assert through
            assert prefix.mass - math.fsum(through) == pytest.approx(below[prefix.destination],
                                                                     abs=1e-7)
        # Exactly: the unexpanded subtrees hold the walks and absorption the
        # budgeted run did not reach.
        absorbed = {kind: full.bias[kind] + full.stabilizer[kind] - cut.bias[kind]
                    - cut.stabilizer[kind] for kind in full.bias}
        missing = (math.fsum(w.relevance for w in full.walks)
                   - math.fsum(w.relevance for w in cut.walks))
        assert math.fsum(p.mass for p in cut.prefixes) == pytest.approx(
            missing + math.fsum(absorbed.values()), abs=1e-12)


@pytest.mark.parametrize('top_k,displayed,omitted', [(1, 2.5, 8.75), (3, 6.25, 5.0),
                                                      (8, 11.25, 0.0)])
def test_top_k_display_omissions_by_hand(top_k, displayed, omitted):
    module, batch = b1(gnn_bias=True, head_bias=-0.25)
    (row, ) = rows_for(module, batch, [config(gamma=0.25, top_k=top_k, feature_vectors=True)])
    ledger = row['ledger']
    assert row['status'] == 'available'
    assert row['coverage']['status'] == ('complete' if top_k == 8 else 'complete_top_k')
    assert ledger['displayed_walk_relevance'] == pytest.approx(displayed * PHI, abs=1e-7)
    assert ledger['omitted_display_relevance'] == pytest.approx(omitted * PHI, abs=1e-7)
    assert ledger['omitted_display_abs_relevance'] == pytest.approx(omitted * PHI, abs=1e-7)
    # Source totals stay complete although walks are hidden from display.
    assert math.fsum(s['relevance'] for s in row['sources']) == pytest.approx(
        11.25 * PHI, abs=1e-7)
    assert sum(s['walk_count'] for s in row['sources']) == 8


# Coincidence with the legacy rule and the v1 capture ##############################

def legacy_case(seed):
    options = dict(T=3 if seed % 6 == 5 else 2, pre=seed % 2,
                   head_layers=1 + (seed // 2) % 2,
                   aggr='mean' if (seed // 4) % 2 == 0 else 'sum',
                   learn_skip=(seed // 8) % 2 == 0, gnn_bias=seed % 3 != 0,
                   head_bias=seed % 5 != 0)
    zero_rows = (('transactions', 1), ) if seed % 4 == 1 else ()
    return options, dict(T=options['T'], zero_rows=zero_rows)


@pytest.mark.parametrize('seed', range(24))
def test_equals_the_legacy_rule_on_nonnegative_inputs(seed):
    """With ReLU and nonnegative inputs every rule input is nonnegative, so
    `x * w_hat = (x * w)_hat` and both rules agree walk by walk, vector by
    vector and kind by kind (contract D5: a null comparison on Rel-HM)."""
    options, graph = legacy_case(seed)
    module = as_binary(relhm_module(seed, **options))
    batch = relhm_batch(seed, **graph)
    for gamma in (0.0, 0.25, 1.0):
        for label in (True, False):
            common = dict(gamma=gamma, top_k=10000, feature_vectors=True,
                          selected_class=label)
            new, old = rows_for(module, batch, [config(**common),
                                                WalkAttributionConfig(**common)])
            assert new['status'] == old['status'] == 'available', (new.get('reason'),
                                                                   old.get('reason'))
            tol = 1e-12 * max(1.0, mass(new))
            a, b = ({row_key(w): w for w in r['walks']} for r in (new, old))
            assert set(a) == set(b)
            for key in a:
                assert abs(a[key]['relevance'] - b[key]['relevance']) <= tol
                assert all(abs(p - q) <= tol for p, q in zip(a[key]['input_relevance'],
                                                             b[key]['input_relevance']))
            for s, t in zip(new['sources'], old['sources']):
                assert (s['table'], s['occurrence']) == (t['table'], t['occurrence'])
                assert abs(s['relevance'] - t['relevance']) <= tol
            for item in ('bias_relevance_by_kind', 'stabilizer_residual_by_kind'):
                assert all(abs(new['ledger'][item][k] - old['ledger'][item][k]) <= tol
                           for k in new['ledger'][item])
            assert new['coverage'] == old['coverage'] and new['exactness'] == old['exactness']
            assert new['method_settings'] == {**old['method_settings'], 'gamma_rule': 'weight'}


@pytest.mark.parametrize('seed', range(6))
def test_equals_the_v1_capture_on_nonnegative_inputs(seed):
    """An independent code path: Engine's v1 `WalkCapture` (contribution rule,
    float32-effective epsilon) with head and preprocessing steps dropped."""
    options, graph = legacy_case(seed)
    module = relhm_module(seed, **options)
    batch = relhm_batch(seed, **graph)
    pre, T = module.model.num_pre_mp_layers, module.model.num_mp_layers
    for gamma in (0.0, 0.25, 1.0):
        with WalkCapture(module, WalkAttributionConfig(gamma=gamma, top_k=10 ** 6,
                                                       max_walks=10 ** 6)) as cap:
            forward(module, batch)
            result = cap.result()
        v1 = {}
        for walk in result['walks']:
            key = tuple((s['layer'] - pre + 1, 'relation', *s['type'], s['edge_occurrence'])
                        if s['kind'] == 'relation' else
                        (s['layer'] - pre + 1, 'self_update', s['table'], '', s['table'],
                         s['occurrence'])
                        for s in walk['steps'] if pre <= s['layer'] < pre + T)
            v1[key] = walk['relevance']
        (row, ) = rows_for(module, batch, [config(gamma=gamma, top_k=10000)])
        walks = walks_of(row)
        assert set(walks) == set(v1)
        tol = 1e-12 * max(1.0, mass(row))
        assert all(abs(walks[key] - v1[key]) <= tol for key in v1)
        assert row['ledger']['bias_relevance'] == pytest.approx(result['bias_relevance'],
                                                                abs=tol)


def test_signed_inputs_separate_the_two_rules():
    """With identity activations and signed inputs the rules differ by
    definition; both ledgers close (n4: neither form is validated there)."""
    module = relhm_module(5, act=None, head_layers=2)
    batch = relhm_batch(5, signed=True)
    for gamma in (0.25, 1.0):
        new, old = rows_for(module, batch, [config(gamma=gamma, top_k=10000),
                                            WalkAttributionConfig(gamma=gamma, top_k=10000)])
        assert new['status'] == old['status'] == 'available'
        a, b = walks_of(new), walks_of(old)
        assert max(abs(a[key] - b[key]) for key in a) > 1e-3


# The forward-rewrite oracle ######################################################

ORACLE_CASES = {
    'b1_bias': lambda: (b1(gnn_bias=True, head_bias=-0.25)),
    'mean_biases': lambda: (relhm_module(1), relhm_batch(1)),
    'sum_identity_root': lambda: (relhm_module(2, aggr='sum', learn_skip=False),
                                  relhm_batch(2)),
    'preprocessing_two_head_layers': lambda: (relhm_module(3, pre=1, head_layers=2),
                                              relhm_batch(3)),
    'bias_free': lambda: (relhm_module(4, gnn_bias=False, head_bias=False), relhm_batch(4)),
    'signed_identity_activation': lambda: (relhm_module(5, act=None, head_layers=2),
                                           relhm_batch(5, signed=True)),
    'signed_inputs_relu': lambda: (relhm_module(6), relhm_batch(6, signed=True)),
    'zero_rows_and_channels': lambda: (relhm_module(7), relhm_batch(
        7, zero_rows=[('article', 0)], zero_channels=[('transactions', 1)])),
    't3_mean_biases': lambda: (relhm_module(8, T=3, head_layers=2), relhm_batch(8, T=3)),
    't3_signed_identity': lambda: (relhm_module(9, T=3, act=None, learn_skip=False, pre=1),
                                   relhm_batch(9, T=3, signed=True)),
    't3_sum_bias_free': lambda: (relhm_module(10, T=3, aggr='sum', gnn_bias=False,
                                              head_bias=False), relhm_batch(10, T=3)),
}


@pytest.mark.parametrize('name', sorted(ORACLE_CASES))
def test_rows_equal_the_forward_rewrite_oracle(name):
    """Walk scores, per-source vectors, and bias and stabilizer by kind from
    the paper's forward rewrite with autograd (independent code), for both
    classes (sign folding) and gamma 0, 0.25 and 1."""
    module, batch = ORACLE_CASES[name]()
    module = as_binary(module)
    for gamma in (0.0, 0.25, 1.0):
        rows = rows_for(module, batch, [config(gamma=gamma, top_k=10000, feature_vectors=True,
                                               selected_class=label)
                                        for label in (True, False)])
        for row in rows:
            assert row['status'] == 'available', row.get('reason')
            sign = row['output']['binary']['logit_sign']
            expected = oracle.explain(module, batch, gamma=gamma, sign=sign)
            tol = 1e-12 * max(1.0, mass(row))
            walks = walks_of(row)
            assert set(walks) == set(expected['walks'])
            assert all(abs(walks[k] - expected['walks'][k]) <= tol for k in walks)
            for source in row['sources']:
                vector = expected['per_source'][(source['table'], source['occurrence'])]
                assert all(abs(p - q) <= tol for p, q in zip(source['input_relevance'], vector))
            ledger = row['ledger']
            for kind in ('preprocessing', 'message_passing', 'head'):
                assert abs(ledger['bias_relevance_by_kind'][kind]
                           - expected['bias'][kind]) <= tol
                assert abs(ledger['stabilizer_residual_by_kind'][kind]
                           - expected['stabilizer'][kind]) <= tol
            assert abs(expected['score'] - row['score']) <= 1e-12 * max(1.0, abs(row['score']))


def test_t3_walks_repeat_edges_and_match_the_oracle():
    module, batch = relhm_module(11, T=3), relhm_batch(11, T=3, n=2, k=2, m=2)
    (row, ) = rows_for(module, batch, [config(gamma=0.25, top_k=10000)])
    walks = walks_of(row)
    assert row['coverage']['total_walks'] == 25 == len(walks)
    repeated = [key for key in walks
                if len({step[1:] for step in key}) < len(key)]
    assert any(key[0][3] == FK[1] and key[2][3] == FK[1] for key in repeated)
    expected = oracle.explain(module, batch, gamma=0.25)['walks']
    tol = 1e-12 * max(1.0, mass(row))
    assert all(abs(walks[key] - expected[key]) <= tol for key in walks)


def test_parallel_edges_stay_distinct_walks():
    """A repeated FK edge between the same occurrences gives two walks
    (route masks); the paper's node masks would sum them."""
    module, batch = relhm_module(12), relhm_batch(12, duplicate_fk=True)
    (row, ) = rows_for(module, batch, [config(gamma=0.25, top_k=10000)])
    walks = walks_of(row)
    routes = oracle.explain(module, batch, gamma=0.25)['walks']
    nodes = oracle.explain(module, batch, gamma=0.25, masks='node')['walks']
    tol = 1e-12 * max(1.0, mass(row))
    assert all(abs(walks[key] - routes[key]) <= tol for key in walks)
    first, last = (rel(1, FK, 0), SELF_C2), (rel(1, FK, 3), SELF_C2)
    assert walks[first] == pytest.approx(walks[last], abs=tol)
    assert nodes[first] == pytest.approx(walks[first] + walks[last], abs=tol)


def test_oracle_forward_and_masks_are_self_consistent():
    for module, batch in (b1(gnn_bias=True, head_bias=-0.25), (relhm_module(8, T=3, pre=1),
                                                               relhm_batch(8, T=3))):
        expected = float(forward(module, batch).detach().reshape(-1)[0])
        trace = oracle.forward_trace(module, *batch)
        assert abs(float(trace.z[0]) - expected) <= 1e-12
        routes = oracle.explain(module, batch, gamma=0.25)['walks']
        nodes = oracle.explain(module, batch, gamma=0.25, masks='node')['walks']
        assert routes == pytest.approx(nodes, abs=1e-12)


def test_oracle_imports_no_explanation_code():
    tree = ast.parse(Path(oracle.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or '')
        elif isinstance(node, ast.Call) and getattr(node.func, 'id', '') == '__import__':
            pytest.fail('dynamic imports are not allowed in the oracle')
    assert imported and not any(name.startswith('kumo_core.explain') for name in imported)
    assert 'importlib' not in imported


# Identity roots, folding, vectors and inactive neurons ############################

@pytest.mark.parametrize('signed', [False, True])
def test_identity_root_is_the_weight_identity(signed):
    """`learn_skip=False` equals learned roots with `W = I` and zero bias,
    under both rules, and the identity route contributes `(1 + gamma) x`."""
    act = None if signed else 'relu'
    identity = relhm_module(13, learn_skip=False, act=act)
    learned = relhm_module(13, learn_skip=True, act=act)
    with torch.no_grad():
        parameters = dict(learned.named_parameters())
        for name, param in identity.named_parameters():
            parameters[name].copy_(param)
        for root in learned.model.root_skips:
            for lin in root.lin_dict.values():
                lin.weight.copy_(torch.eye(3, dtype=F64))
                lin.bias.zero_()
    batch = relhm_batch(13, signed=signed)
    for method in (WEIGHT, LEGACY_METHOD):
        for gamma in (0.0, 0.25, 1.0):
            cfg = WalkAttributionConfig(method=method, gamma=gamma, top_k=10000)
            (a, ) = rows_for(identity, batch, [cfg])
            (b, ) = rows_for(learned, batch, [cfg])
            tol = 1e-13 * max(1.0, mass(a))
            walks_a, walks_b = walks_of(a), walks_of(b)
            assert set(walks_a) == set(walks_b)
            assert all(abs(walks_a[key] - walks_b[key]) <= tol for key in walks_a)
    computation = computation_of(identity, batch)
    rule = weight_gamma.make_rule(computation, Selection('regression', 0, 1, None),
                                  config(gamma=0.25))
    (route, *_) = computation.incoming(1, ('customer', 0)).routes
    x = computation.captured_input(1)['customer'][0]
    assert route.operator == 'identity'
    assert torch.equal(rule.modified_contribution(x, route.weight, 1), torch.diag(1.25 * x))


def test_folding_is_negation_only_at_gamma_zero():
    module, batch = as_binary(relhm_module(14)), relhm_batch(14)
    for gamma in (0.0, 0.25):
        positive, negative = rows_for(module, batch, [
            config(gamma=gamma, top_k=10000, feature_vectors=True, selected_class=label)
            for label in (True, False)])
        a, b = walks_of(positive), walks_of(negative)
        assert negative['score'] == -positive['score']
        if gamma == 0.0:
            assert all(b[key] == -a[key] for key in a)
            assert all(s['input_relevance'] == [-v for v in t['input_relevance']]
                       for s, t in zip(negative['sources'], positive['sources']))
        else:
            assert max(abs(b[key] + a[key]) for key in a) > 1e-6


def test_vectors_sources_and_exact_zeros():
    """Walk vectors live at `H_0` and sum to the walk; channels and rows with
    zero input carry exactly zero relevance; source totals cover every walk,
    displayed or not, and equal the unmasked oracle."""
    module = relhm_module(15, pre=1)
    batch = relhm_batch(15, zero_rows=[('transactions', 2)],
                        zero_channels=[('article', 0), ('customer', 2)])
    (row, ) = rows_for(module, batch, [config(gamma=0.25, top_k=3, feature_vectors=True)])
    assert row['status'] == 'available' and row['coverage']['status'] == 'complete_top_k'
    for walk in row['walks']:
        source = next(s for s in row['sources']
                      if (s['table'], s['occurrence']) == (walk['source_table'],
                                                           walk['source_occurrence']))
        assert math.fsum(walk['input_relevance']) == pytest.approx(walk['relevance'],
                                                                   abs=1e-12)
        assert all(r == 0.0 for r, v in zip(walk['input_relevance'], source['input_value'])
                   if v == 0.0)
    expected = oracle.explain(module, batch, gamma=0.25)['per_source']
    tol = 1e-12 * max(1.0, mass(row))
    zero_row = next(s for s in row['sources'] if (s['table'], s['occurrence'])
                    == ('transactions', 2))
    assert zero_row['input_relevance'] == [0.0] * 3 and zero_row['relevance'] == 0.0
    for source in row['sources']:
        assert source['input_value'] == batch[0][source['table']][source['occurrence']].tolist()
        assert all(r == 0.0 for r, v in zip(source['input_relevance'], source['input_value'])
                   if v == 0.0)
        assert all(abs(p - q) <= tol for p, q in zip(
            source['input_relevance'], expected[(source['table'], source['occurrence'])]))
    assert sum(s['walk_count'] for s in row['sources']) == row['coverage']['total_walks']
    assert sum(s['displayed_walk_count'] for s in row['sources']) == 3


def test_inactive_relu_neurons_receive_exactly_zero_relevance():
    """Every child share at a source neuron whose captured value is 0 is an
    exact zero, at every expanded destination and layer."""
    seen = []

    class Recording(weight_gamma.WeightGammaRule):
        def transfer(self, layer, destination, state, incoming):
            children, absorbed = super().transfer(layer, destination, state, incoming)
            inputs = self.computation.captured_input(layer)
            for route, child in zip(incoming.routes, children):
                x = inputs[route.source[0]][route.source[1]]
                seen.extend(float(c) for c, v in zip(child, x) if v == 0)
            return children, absorbed

    module, batch = relhm_module(16, T=3, pre=1), relhm_batch(16, T=3)
    _, _, enumeration = enumerate_rule(module, batch, config(gamma=0.25),
                                       rule=lambda c, s, cfg: Recording(c, s, cfg))
    assert enumeration.complete and len(seen) > 10 and all(c == 0.0 for c in seen)


# Accounting #####################################################################

def test_a_dropped_share_fails_the_row():
    """The ledger is not tautological: a copy of the rule that drops one
    route's share is caught as `accounting_error_exceeded`."""
    class Dropping(weight_gamma.WeightGammaRule):
        def transfer(self, layer, destination, state, incoming):
            children, absorbed = super().transfer(layer, destination, state, incoming)
            return children[:-1] + [torch.zeros_like(children[-1])], absorbed

    module, batch = b1(gnn_bias=True, head_bias=-0.25)
    cfg = config(gamma=0.25)
    rows = {}
    for name, make in (('real', weight_gamma.make_rule),
                       ('dropping', lambda c, s, f: Dropping(c, s, f))):
        computation, selection, enumeration = enumerate_rule(module, batch, cfg, rule=make)
        logits = forward(module, batch)
        rows[name] = build_row(
            computation, enumeration, weight_gamma.summarize(computation, selection, cfg,
                                                              enumeration),
            config=cfg, selection=selection, output=output_block(selection, module, logits[0]),
            qualification=list(STRUCTURAL_CODES), architecture=_architecture(module),
            operations=_operations(module), sample=sample_of(batch),
            origin='original_forward_attribution', prediction_id=ID, sample_sha256=SAMPLE_ID,
            numerics=row_numerics(NUMERICS, 'float64'))
    assert rows['real']['status'] == 'available'
    validate_v2_row(rows['real'], sample_of(batch))
    assert rows['dropping']['status'] == 'failed'
    assert rows['dropping']['reason'] == 'accounting_error_exceeded'


def _architecture(module):
    from kumo_core.explain.walk_graph import architecture_record
    return architecture_record(module)


def _operations(module):
    from kumo_core.explain.walk_graph import operations_record
    return operations_record(module)


@pytest.mark.parametrize('gnn_bias,head_bias,budget', [
    (False, None, None), (True, -0.25, None), (False, 0.5, None), (True, None, 6),
    (False, None, 6)])
def test_bias_kinds_and_reasons_follow_actual_parameters(gnn_bias, head_bias, budget):
    module, batch = b1(gnn_bias=gnn_bias, head_bias=head_bias)
    options = {} if budget is None else {'max_walks': budget}
    (row, ) = rows_for(module, batch, [config(gamma=0.25, **options)])
    biased = gnn_bias or head_bias is not None
    reasons = ['method_is_lrp'] + (['biases_present'] if biased else []) + (
        [] if budget is None else ['enumeration_incomplete'])
    assert row['exactness']['reasons'] == reasons
    assert row['exactness']['decomposition'] == 'conservative_lrp'
    assert not row['exactness']['degree_t_claim']
    by_kind = row['ledger']['bias_relevance_by_kind']
    assert by_kind['preprocessing'] == 0.0
    assert (by_kind['message_passing'] == 0.0) is not gnn_bias
    assert (by_kind['head'] == 0.0) is (head_bias is None)
    assert row['ledger']['precision_residual'] == 0.0
    assert row['method_settings'] == {
        'gamma': 0.25, 'gamma_rule': 'weight', 'bias_rule': 'gamma_on_constant_input',
        'epsilon': 1e-9, 'stabilizer': 'sign_preserving_epsilon',
        'identity_root_rule': 'identity_as_weight', 'sign_folding': 'selected_output_row'}


def test_rows_are_deterministic_and_ignore_edge_type_insertion_order():
    module, _ = b1(gnn_bias=True, head_bias=-0.25)
    rows = []
    for order in (None, [ART_FK, ART_REV, REV, FK], None):
        batch = (*b1_batch(order=order), )
        rows.append(rows_for(module, batch, [config(gamma=0.25, feature_vectors=True)])[0])
    assert rows[0] == rows[1] == rows[2]


def test_failure_paths():
    # A nonfinite parameter fails the row (here the model output is infinite).
    name = 'model.hetero_convs.0.<transactions___customer_id___customer>.lin.weight'
    module = b1_module(values={name: [[math.inf]]})
    (row, ) = rows_for(module, (*b1_batch(), ), [config(gamma=0.25)])
    assert row['status'] == 'failed' and 'walks' not in row
    # A gather destination with two incoming edges is unsupported.
    module, (x, edges, nodes, counts) = b1()
    x = {**x, 'article': tensor([[4.0], [5.0]])}
    edges = {**edges, ART_REV: torch.tensor([[0, 1], [0, 0]])}
    nodes = {**nodes, 'article': [0, 0, 2]}
    counts = {**counts, ART_REV: [0, 2]}
    (row, ) = rows_for(module, (x, edges, nodes, counts), [config()])
    assert row['status'] == 'unsupported' and 'multiple incoming' in row['reason']
    # Duplicate transactions with equal features and neighborhoods stay
    # distinct occurrences with walks of equal relevance.
    module, (x, edges, nodes, counts) = b1(gnn_bias=True, head_bias=-0.25)
    x = {**x, 'transactions': tensor([[2.0], [2.0]])}
    edges = {**edges, ART_REV: torch.tensor([[0, 0], [0, 1]])}
    counts = {**counts, ART_REV: [0, 2]}
    (row, ) = rows_for(module, (x, edges, nodes, counts), [config(gamma=0.25, top_k=100)])
    walks = walks_of(row)
    for first, second in (((rel(1, FK, 0), SELF_C2), (rel(1, FK, 1), SELF_C2)),
                          ((rel(1, ART_REV, 0), rel(2, FK, 0)),
                           (rel(1, ART_REV, 1), rel(2, FK, 1)))):
        assert walks[first] == pytest.approx(walks[second], rel=1e-15)
    assert row['coverage']['total_walks'] == 9


# Diagnostics ####################################################################

@pytest.mark.parametrize('seed', range(8))
def test_no_denominator_flips_for_positive_scores_on_nonnegative_inputs(seed):
    """Below the head of a ReLU model with nonnegative inputs, `S >= a > 0` at
    every relevant neuron (brainstorm 4.1); at the output, for a positive
    selected logit. Float64 captures have no precision mismatch."""
    options, graph = legacy_case(seed)
    module, batch = relhm_module(seed, **options), relhm_batch(seed, **graph)
    z = float(forward(module, batch).reshape(-1)[0])
    for gamma in (0.0, 0.25, 1.0):
        computation, selection, enumeration = enumerate_rule(
            module, batch, config(gamma=gamma), sign=1 if z > 0 else -1)
        diagnostics = weight_gamma.summarize(computation, selection, config(gamma=gamma),
                                             enumeration)['diagnostics']
        assert diagnostics['denominator_sign_flips'] == 0
        assert not diagnostics['head_output_flipped']
        assert diagnostics['relevant_neurons'] > 1
        assert 0.0 < diagnostics['min_denominator_margin'] <= 1.0
        assert diagnostics['max_preactivation_mismatch'] <= 1e-13


def test_identity_activation_can_flip_a_hidden_denominator():
    """With identity activations a negative hidden value can get a positive
    gamma-modified total: root row (3, -4) on x = (1, 1) gives a = -1 but
    S = 2 at gamma 1."""
    module, batch = tiny([1.0, 1.0], [1.0, 2.0], root=[[3.0, -4.0], [1.0, 0.0]])
    computation, selection, enumeration = enumerate_rule(module, batch, config(gamma=1.0))
    diagnostics = weight_gamma.denominator_diagnostics(computation, selection,
                                                       config(gamma=1.0))
    assert diagnostics['denominator_sign_flips_by_kind'] == {
        'preprocessing': 0, 'message_passing': 1, 'head': 0}
    assert diagnostics['relevant_neurons'] == 3 and not diagnostics['head_output_flipped']
    plain = weight_gamma.denominator_diagnostics(computation, selection, config(gamma=0.0))
    assert plain['denominator_sign_flips'] == 0


def test_diagnostics_are_static_and_independent_of_the_budget():
    module, batch = b1(head_bias=-9.5)
    complete, budgeted = rows_for(module, batch, [config(gamma=0.25),
                                                  config(gamma=0.25, max_walks=3)])
    assert budgeted['status'] == 'incomplete'
    assert complete['diagnostics'] == budgeted['diagnostics']
    assert complete['diagnostics'][WEIGHT]['relevant_neurons'] == 5


def test_zero_score_has_no_relevant_output():
    module, batch = tiny([1.0, 2.0], [2.0, -1.0], act='relu')
    (row, ) = rows_for(module, batch, [config(gamma=0.25)])
    assert row['score'] == 0.0 and row['status'] == 'available'
    diagnostics = row['diagnostics'][WEIGHT]
    assert diagnostics['head_output_flipped'] is False
    assert diagnostics['relevant_neurons'] == 2  # the two hidden values, not the output
    assert diagnostics['denominator_sign_flips_by_kind']['head'] == 0


def test_inactive_neurons_are_not_relevant_positions():
    """A ReLU value of 0 carries no relevance, so its denominator is not
    checked: x = (1, -1) gives H = (1, 0) and two relevant neurons."""
    module, batch = tiny([1.0, -1.0], [2.0, 3.0], act='relu')
    computation, selection, enumeration = enumerate_rule(module, batch, config(gamma=0.25))
    diagnostics = weight_gamma.summarize(computation, selection, config(gamma=0.25),
                                         enumeration)['diagnostics']
    assert diagnostics['relevant_neurons'] == 2
    assert diagnostics['min_denominator_margin'] == 1.0
    assert diagnostics['head_output_denominator'] == 2.5


# Row validation ##################################################################

def mutate(row, change):
    broken = copy.deepcopy(row)
    change(broken)
    return broken


def _diagnostics(row):
    return row['diagnostics'][WEIGHT]


MUTATIONS = {
    'contribution_rule': lambda r: r['method_settings'].update(gamma_rule='contribution'),
    'gamma_setting': lambda r: r['method_settings'].update(gamma=0.5),
    'precision_residual': lambda r: r['ledger'].update(
        precision_residual=1e-12, accounting_error=r['ledger']['accounting_error'] - 1e-12),
    'degree_t_claim': lambda r: r['exactness'].update(degree_t_claim=True, reasons=[]),
    'missing_biases_present': lambda r: r['exactness'].update(reasons=['method_is_lrp']),
    'reason_order': lambda r: r['exactness'].update(
        reasons=['biases_present', 'method_is_lrp']),
    'extra_qualification': lambda r: r['exactness']['qualification'].append('interior_piece'),
    'no_diagnostics': lambda r: r.update(diagnostics=None),
    'diagnostic_key': lambda r: _diagnostics(r).pop('min_denominator_margin'),
    'flip_without_count': lambda r: _diagnostics(r).update(head_output_flipped=True),
    'denominator_contradicts_flag': lambda r: _diagnostics(r).update(
        head_output_denominator=-1.0),
    'flip_count_type': lambda r: _diagnostics(r).update(denominator_sign_flips=1.0),
    'flips_do_not_add_up': lambda r: _diagnostics(r).update(denominator_sign_flips=3),
    'margin_above_one': lambda r: _diagnostics(r).update(min_denominator_margin=1.5),
    'negative_mismatch': lambda r: _diagnostics(r).update(max_preactivation_mismatch=-1.0),
}


@pytest.mark.parametrize('name', sorted(MUTATIONS))
def test_validator_rejects_method_mutations(name):
    module, batch = b1(gnn_bias=True, head_bias=-0.25)
    (row, ) = rows_for(module, batch, [config(gamma=0.25, feature_vectors=True)])
    validate_v2_row(json.loads(json.dumps(row, allow_nan=False)), sample_of(batch))
    with pytest.raises(WalkSchemaError):
        validate_v2_row(mutate(row, MUTATIONS[name]), sample_of(batch))


def test_validator_rejects_bias_relevance_without_biases():
    module, batch = b1()
    (row, ) = rows_for(module, batch, [config(gamma=0.25)])
    broken = mutate(row, lambda r: (
        r['ledger']['bias_relevance_by_kind'].update(head=0.5),
        r['ledger'].update(bias_relevance=0.5, stabilizer_residual=r['ledger'][
            'stabilizer_residual'] - 0.5),
        r['ledger']['stabilizer_residual_by_kind'].update(
            head=r['ledger']['stabilizer_residual_by_kind']['head'] - 0.5)))
    with pytest.raises(WalkSchemaError, match='No head bias'):
        weight_gamma.validate_row(broken)
    with pytest.raises(WalkSchemaError, match='Malformed'):
        weight_gamma.validate_row({'method': WEIGHT})
