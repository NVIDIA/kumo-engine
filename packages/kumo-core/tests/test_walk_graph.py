# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed capture: fidelity, trimming, routes, masks, numerics and copies.

The builders at the top are shared by the other core walk test modules.
"""
import copy

import pytest
import torch

from kumo_core.explain.walk_attribution import (
    UnsupportedWalkAttribution,
    WalkAttributionConfig,
    WalkCapture,
)
from kumo_core.explain.walk_graph import (
    GraphCapture,
    architecture_record,
    capture_numerics,
    float32_matmul_precision,
    mkldnn_matmul_fp32_precision,
    normalized_mkldnn,
    operations_record,
    structural_qualification,
)
from kumo_core.modeling.gnn.hetero import (
    ForeignKeyToPrimaryKeyConv,
    HeteroGNN,
)
from kumo_core.modeling.heads.node_prediction import NodePredHead

F64 = torch.float64
FK = ('transactions', 'customer_id', 'customer')
REV = ('customer', 'rev_customer_id', 'transactions')
ART_REV = ('article', 'rev_article_id', 'transactions')
ART_FK = ('transactions', 'article_id', 'article')
TABLES = ['customer', 'transactions', 'article']


# Shared builders ############################################################

def b1_module(dtype=F64, *, gnn_bias=False, head_bias=False, act='relu',
              learn_skip=True, aggr='mean', head_layers=1, values=None):
    """P0 fixture B1 (all weights 1); `values` overrides parameters by name."""
    module = torch.nn.Module()
    module.entity_table_name = 'customer'
    module.train_table_name = 'task'
    module.model = HeteroGNN(1, 2, metadata=(TABLES, [FK, REV, ART_REV, ART_FK]),
                             aggr=[aggr], norm=None, bias=gnn_bias,
                             learn_skip=learn_skip, act=act)
    module.head = NodePredHead(1, 1, num_layers=head_layers, norm=None,
                               param_keys=[], bias=head_bias)
    with torch.no_grad():
        for name, param in module.named_parameters():
            param.fill_(0.5 if name.endswith('bias') else 1.0)
            if values and name in values:
                param.copy_(torch.as_tensor(values[name], dtype=param.dtype))
    return module.to(dtype).eval()


def b1_batch(dtype=F64, order=None, x=None):
    x = x or {'customer': [[1.]], 'transactions': [[2.], [3.]], 'article': [[4.]]}
    x = {k: torch.tensor(v, dtype=dtype) for k, v in x.items()}
    edges = {FK: torch.tensor([[0, 1], [0, 0]]), REV: torch.tensor([[0, 0], [0, 1]]),
             ART_REV: torch.tensor([[0], [0]]), ART_FK: torch.zeros(2, 0, dtype=torch.long)}
    if order is not None:
        edges = {key: edges[key] for key in order}
    nodes = {'customer': [1, 0, 0], 'transactions': [0, 2, 0], 'article': [0, 0, 1]}
    nse = {FK: [2, 0], REV: [0, 2], ART_REV: [0, 1], ART_FK: [0, 0]}
    return x, edges, nodes, nse


def b1_forward(module, batch):
    x, edges, nodes, nse = batch
    out = module.model(dict(x), dict(edges), {k: list(v) for k, v in nodes.items()},
                       {k: list(v) for k, v in nse.items()})
    return module.head(out[module.entity_table_name][:1])


def capture(module, run, config=None):
    with GraphCapture(module, config) as cap:
        output = run()
    return cap, output


def toy_t3_module(dtype=F64):
    """customer c0 and transaction t0, FK and REV edges, T = 3, no trimming.

    All destinations exist at every layer, so walks repeat variables and
    reorder them (multi-walk bags).
    """
    module = torch.nn.Module()
    module.entity_table_name = 'customer'
    module.train_table_name = 'task'
    module.model = HeteroGNN(1, 3, metadata=(['customer', 'transactions'], [FK, REV]),
                             aggr=['sum'], norm=None, bias=False, learn_skip=True)
    module.head = NodePredHead(1, 1, num_layers=1, norm=None, param_keys=[], bias=False)
    with torch.no_grad():
        for param in module.parameters():
            param.fill_(1.0)
    module = module.to(dtype).eval()
    x = {'customer': torch.tensor([[1.]], dtype=dtype),
         'transactions': torch.tensor([[2.]], dtype=dtype)}
    edges = {FK: torch.tensor([[0], [0]]), REV: torch.tensor([[0], [0]])}
    return module, lambda: module.head(module.model(dict(x), dict(edges))['customer'][:1])


def preprocessing_module(dtype=F64, act='relu'):
    """One pre-MP layer, learned roots, two head layers and signed biases."""
    relation = ('events', 'entity_id', 'entities')
    module = torch.nn.Module()
    module.entity_table_name = 'entities'
    module.train_table_name = 'task'
    module.model = HeteroGNN(channels=2, num_mp_layers=1,
                             metadata=(['events', 'entities'], [relation]),
                             num_pre_mp_layers=1, aggr='mean', act=act, norm=None,
                             bias=True, learn_skip=True)
    module.head = NodePredHead(in_channels=2, out_channels=1, hidden_channels=2,
                               num_layers=2, norm=None, param_keys=[], act=act)
    values = {
        'model.lins.0.lin_dict.events.weight': [[1., -0.5], [0.25, 1.]],
        'model.lins.0.lin_dict.events.bias': [0.5, -0.25],
        'model.lins.0.lin_dict.entities.weight': [[0.5, 0.5], [-1., 1.]],
        'model.lins.0.lin_dict.entities.bias': [0.125, 0.25],
        'model.root_skips.0.lin_dict.events.weight': [[0.5, 0.], [0., 0.5]],
        'model.root_skips.0.lin_dict.events.bias': [0., 0.125],
        'model.root_skips.0.lin_dict.entities.weight': [[1., -0.25], [0.5, 0.75]],
        'model.root_skips.0.lin_dict.entities.bias': [-0.125, 0.5],
        'head.mlp.lins.0.weight': [[1., -1.], [0.5, 0.5]],
        'head.mlp.lins.0.bias': [0.5, -0.5],
        'head.mlp.lins.1.weight': [[2., -1.]],
        'head.mlp.lins.1.bias': [0.25],
    }
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name in values:
                param.copy_(torch.tensor(values[name]))
            elif 'hetero_convs' in name:
                param.copy_(torch.tensor([[1., 0.5], [-0.5, 1.]] if name.endswith('weight')
                                         else [0.25, -0.5]))
    module = module.to(dtype).eval()
    x = {'events': torch.tensor([[1., 2.], [0.5, 1.5], [2., 0.]], dtype=dtype),
         'entities': torch.tensor([[1., 0.5]], dtype=dtype)}
    edges = {relation: torch.tensor([[0, 1, 2], [0, 0, 0]])}
    return module, lambda: module.head(module.model(dict(x), dict(edges))['entities'], None)


def random_b1(seed, dtype=F64, **options):
    """B1 topology with random signed weights and 3-channel features."""
    generator = torch.Generator().manual_seed(seed)
    module = torch.nn.Module()
    module.entity_table_name = 'customer'
    module.train_table_name = 'task'
    module.model = HeteroGNN(3, 2, metadata=(TABLES, [FK, REV, ART_REV, ART_FK]),
                             aggr=[options.get('aggr', 'mean')], norm=None,
                             bias=options.get('gnn_bias', True),
                             learn_skip=options.get('learn_skip', True),
                             num_pre_mp_layers=options.get('pre', 0),
                             act=options.get('act', 'relu'))
    module.head = NodePredHead(3, 1, num_layers=options.get('head_layers', 2), norm=None,
                               param_keys=[], bias=options.get('head_bias', True))
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=generator))
    module = module.to(dtype).eval()
    x = {k: torch.rand(n, 3, generator=generator, dtype=dtype)
         for k, n in (('customer', 1), ('transactions', 2), ('article', 1))}
    batch = b1_batch(dtype, x={k: v.tolist() for k, v in x.items()})
    return module, lambda: b1_forward(module, batch)


# Fidelity ###################################################################

@pytest.mark.parametrize('dtype', [torch.float32, F64])
def test_capture_is_bit_identical_and_removes_hooks(dtype):
    module, batch = b1_module(dtype), b1_batch(dtype)
    expected = b1_forward(module, batch)
    cap, actual = capture(module, lambda: b1_forward(module, batch))
    assert torch.equal(actual, expected)
    assert not cap.handles
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in module.modules())
    computation = cap.computation()
    assert computation.head_output.tolist() == expected.reshape(-1).tolist()
    assert computation.model_dtype == str(dtype).removeprefix('torch.')


def test_hooks_are_removed_after_an_exception():
    module = b1_module()
    with pytest.raises(RuntimeError, match='injected'):
        with GraphCapture(module):
            raise RuntimeError('injected')
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in module.modules())


def test_second_forward_marks_the_capture_failed_but_keeps_predictions():
    module, batch = b1_module(), b1_batch()
    with GraphCapture(module) as cap:
        first = b1_forward(module, batch)
        second = b1_forward(module, batch)
    assert torch.equal(first, second)
    with pytest.raises(ValueError, match='capture failed'):
        cap.computation()


def test_trimming_destinations_and_frozen_layer_denominators():
    module, batch = b1_module(), b1_batch()
    cap, _ = capture(module, lambda: b1_forward(module, batch))
    computation = cap.computation()
    assert computation.T == 2 and computation.seed == ('customer', 0)
    assert computation.destinations(1) == {'customer': 1, 'transactions': 2, 'article': 0}
    assert computation.destinations(2) == {'customer': 1, 'transactions': 0, 'article': 0}
    # Layer 2 keeps only the hop-1 FK edges; mean denominators are per layer.
    layer2 = computation.incoming(2, ('customer', 0)).routes
    assert [(r.kind, r.source, r.edge_occurrence, r.denominator) for r in layer2] == [
        ('self_update', ('customer', 0), None, 1),
        ('relation', ('transactions', 0), 0, 2), ('relation', ('transactions', 1), 1, 2)]
    t0 = computation.incoming(1, ('transactions', 0)).routes
    assert [(r.relation, r.operator, r.direction, r.denominator) for r in t0[1:]] == [
        (ART_REV, 'gather', 'pk_to_fk', 1), (REV, 'gather', 'pk_to_fk', 1)]
    assert computation.relevant_nodes(0) == [('article', 0), ('customer', 0),
                                             ('transactions', 0), ('transactions', 1)]
    with pytest.raises(ValueError, match='not a destination'):
        computation.incoming(2, ('transactions', 0))


def test_broken_edge_prefix_fails_the_row():
    module, batch = b1_module(), b1_batch()
    cap, _ = capture(module, lambda: b1_forward(module, batch))
    relation = next(r for r in cap.layers[0].relations if r.relation == REV)
    relation.edge_index = relation.edge_index.flip(1)
    with pytest.raises(ValueError, match='Edge prefix'):
        cap.computation()


def test_route_order_and_keys_ignore_edge_type_insertion_order():
    def routes(order):
        module, batch = b1_module(), b1_batch(order=order)
        cap, _ = capture(module, lambda: b1_forward(module, batch))
        computation = cap.computation()
        return [(t, node, [(r.kind, r.source, r.relation, r.edge_occurrence)
                           for r in computation.incoming(t, node).routes])
                for t in (2, 1) for node in computation.relevant_nodes(t)]

    assert routes(None) == routes([ART_FK, ART_REV, REV, FK])


def test_multi_edge_gather_makes_the_row_unsupported():
    module = b1_module()
    x, edges, nodes, nse = b1_batch(x={'customer': [[1.]], 'transactions': [[2.], [3.]],
                                       'article': [[4.], [5.]]})
    edges[ART_REV] = torch.tensor([[0, 1], [0, 0]])
    nodes['article'] = [0, 0, 2]
    nse[ART_REV] = [0, 2]
    cap, _ = capture(module, lambda: b1_forward(module, (x, edges, nodes, nse)))
    with pytest.raises(UnsupportedWalkAttribution, match='multiple incoming'):
        cap.computation()


def test_direction_and_operator_come_from_the_observed_class():
    module, batch = b1_module(), b1_batch()
    # A `rev_` relation served by a foreign-key convolution is recorded as
    # what it is, not as its name suggests.
    conv = ForeignKeyToPrimaryKeyConv(1, 1, aggr='mean', bias=False).double().eval()
    with torch.no_grad():
        conv.lin.weight.fill_(1.0)
    module.model.hetero_convs[0][ART_REV] = conv
    cap, _ = capture(module, lambda: b1_forward(module, batch))
    route = next(r for r in cap.computation().incoming(1, ('transactions', 0)).routes
                 if r.relation == ART_REV)
    assert (route.direction, route.operator, route.denominator) == ('fk_to_pk', 'mean', 1)
    assert {'relation': list(ART_REV), 'direction': 'fk_to_pk', 'operator': 'mean'} in (
        architecture_record(module)['relations'])


# Piece forward ##############################################################

@pytest.mark.parametrize('build', [
    lambda: (lambda m, b: (m, lambda: b1_forward(m, b)))(b1_module(), b1_batch()),
    preprocessing_module,
    lambda: preprocessing_module(act=None),
    toy_t3_module,
    lambda: random_b1(1),
    lambda: random_b1(2, pre=1, aggr='sum', learn_skip=False),
    lambda: random_b1(3, act=None, head_layers=1),
    lambda: random_b1(4, gnn_bias=False, head_bias=False, pre=1),
])
def test_forward_piece_equals_float64_module_forward(build):
    module, run = build()
    cap, output = capture(module, run)
    computation = cap.computation()
    piece = computation.forward_piece()
    assert abs(float(piece.z_piece[0]) - float(output.detach().reshape(-1)[0])) <= 1e-12
    for t in range(1, computation.T + 1):
        for table, values in piece.hidden[t].items():
            torch.testing.assert_close(values, computation.hidden[t][table], rtol=0,
                                       atol=1e-12)
            torch.testing.assert_close(piece.pre_activation[t][table],
                                       computation.pre_activation[t][table], rtol=0,
                                       atol=1e-12)


def test_piece_exposes_masks_and_activation_kind():
    module, run = preprocessing_module()
    cap, _ = capture(module, run)
    piece = cap.computation().forward_piece()
    assert piece.activation == 'relu'
    assert piece.masks[1]['entities'].dtype == torch.bool
    assert piece.preprocess_masks[0]['events'].shape == (3, 2)
    assert piece.head_masks[0] is not None and piece.head_masks[-1] is None
    identity, run = preprocessing_module(act=None)
    cap, _ = capture(identity, run)
    computation = cap.computation()
    piece = computation.forward_piece()
    assert piece.activation is None and piece.masks[1] is None
    assert piece.preprocess_masks == (None, )
    assert torch.equal(computation.gate_mask(1, ('entities', 0)), torch.ones(2, dtype=F64))


def test_masks_come_from_captured_pre_activations():
    # x_customer = 0 makes the seed's layer-1 pre-activation exactly 0 when
    # its neighbors are switched off; the mask must be 0, not 1.
    values = {'model.hetero_convs.0.<transactions___customer_id___customer>.lin.weight': [[0.]]}
    module = b1_module(values=values)
    batch = b1_batch(x={'customer': [[0.]], 'transactions': [[2.], [3.]], 'article': [[4.]]})
    cap, _ = capture(module, lambda: b1_forward(module, batch))
    computation = cap.computation()
    assert float(computation.pre_activation[1]['customer'][0, 0]) == 0.0
    assert float(computation.gate_mask(1, ('customer', 0))[0]) == 0.0


def test_computation_is_deep_copyable_for_injected_sign_flips():
    module, batch = b1_module(), b1_batch()
    cap, _ = capture(module, lambda: b1_forward(module, batch))
    computation = cap.computation()
    flipped = copy.deepcopy(computation)
    flipped.pre_activation[2]['customer'][0, 0] = -1.0
    assert float(flipped.forward_piece().z_piece[0]) == 0.0
    assert float(computation.forward_piece().z_piece[0]) == 9.0
    assert float(flipped.gate_mask(2, ('customer', 0))[0]) == 0.0


def test_captured_tensors_do_not_alias_live_tensors():
    module, batch = b1_module(), b1_batch()
    cap, _ = capture(module, lambda: b1_forward(module, batch))
    with torch.no_grad():
        batch[0]['transactions'].mul_(10)
        for param in module.parameters():
            param.mul_(3)
    b1_forward(module, batch)
    computation = cap.computation()
    assert computation.h0['transactions'].reshape(-1).tolist() == [2.0, 3.0]
    assert float(computation.forward_piece().z_piece[0]) == 9.0


def test_capture_leaves_parameter_gradients_bitwise_unchanged():
    def gradients(context):
        module, batch = b1_module(torch.float32, gnn_bias=True, head_bias=True), b1_batch(
            torch.float32)
        with context(module):
            out = b1_forward(module, batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out, torch.ones_like(out))
        return torch.autograd.grad(loss, list(module.parameters()), allow_unused=True)

    def same(a, b):
        return (a is None and b is None) or (a is not None and b is not None
                                             and torch.equal(a, b))

    plain = gradients(lambda m: _null())
    typed = gradients(lambda m: GraphCapture(m))
    legacy = gradients(lambda m: WalkCapture(m, WalkAttributionConfig()))
    assert any(g is not None for g in plain)
    assert all(same(a, b) for a, b in zip(plain, typed))
    assert all(same(a, b) for a, b in zip(plain, legacy))


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


# Numerics, qualification and records ########################################

def test_numerics_record_and_precision_context_restore_both_settings():
    record = capture_numerics()
    assert set(record) == {'precision_mode', 'mkldnn_matmul_fp32_precision',
                           'num_threads', 'cpu_capability', 'torch_version'}
    assert normalized_mkldnn('none') == 'ieee' and normalized_mkldnn('bf16') == 'bf16'
    before = (torch.get_float32_matmul_precision(), mkldnn_matmul_fp32_precision())
    with pytest.raises(RuntimeError, match='inside'):
        with float32_matmul_precision('medium'):
            assert torch.get_float32_matmul_precision() == 'medium'
            raise RuntimeError('inside')
    assert (torch.get_float32_matmul_precision(), mkldnn_matmul_fp32_precision()) == before
    with float32_matmul_precision(None):
        assert torch.get_float32_matmul_precision() == before[0]


def test_structural_qualification_raises_before_hooks():
    module = b1_module()
    assert 'calibrator_outside_f' in structural_qualification(module)
    module.model.act = 'gelu'
    with pytest.raises(UnsupportedWalkAttribution):
        with GraphCapture(module):
            pass
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in module.modules())


def test_architecture_record_reads_actual_parameters():
    module = b1_module()
    record = architecture_record(module)
    assert record['gnn_bias'] is False and record['head_bias'] is False
    assert record['relations'] == [
        {'relation': list(ART_REV), 'direction': 'pk_to_fk', 'operator': 'gather'},
        {'relation': list(REV), 'direction': 'pk_to_fk', 'operator': 'gather'},
        {'relation': list(ART_FK), 'direction': 'fk_to_pk', 'operator': 'mean'},
        {'relation': list(FK), 'direction': 'fk_to_pk', 'operator': 'mean'}]
    # A bias attached by hand is reported although the module flag says no.
    conv = module.model.hetero_convs[0][FK]
    conv.lin.bias = torch.nn.Parameter(torch.zeros(1, dtype=F64))
    assert module.model.bias is False and architecture_record(module)['gnn_bias'] is True
    operations = operations_record(preprocessing_module()[0])
    assert operations['preprocessing'] == [{'index': 0, 'kind': 'linear',
                                            'activation': 'relu', 'bias': True}]
    assert [op['activation'] for op in operations['head']] == ['relu', None]
