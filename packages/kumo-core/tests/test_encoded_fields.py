# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fitted field fixtures complement graph-only walk and encoder forward tests."""
from types import SimpleNamespace

import pytest
import torch
from kumo_core.encoders.tensor_frame_encoder import TensorFrameEncoder
from kumo_core.explain.encoded_fields import FieldCapture
from kumo_core.explain.walk_lrp_epsilon import (
    POLICY_ID,
    SMOOTH_FLOOR_POLICY_ID,
)
from kumoapi.encoder import Index, Numerical
from kumoapi.typing import ColStatType, Stype
from torch_frame import TensorFrame


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; GPU field adapter remains unqualified')
    return torch.device('cuda:0' if request.param == 'cuda' else 'cpu')


def fixture(device, *, act=None, category_na='separate'):
    category = Index(na_strategy=category_na)
    category._stats[ColStatType.CATEGORY_COUNTS] = [('a', 3), ('b', 2)]
    encoder = TensorFrameEncoder(
        'transactions', {
            'amount': Numerical(scaler=None, na_strategy='separate'),
            'category': category
        }, None, 1, act=act, device=device)
    numerical = encoder.stype_encoder_dict[Stype.numerical.value]
    categorical = encoder.stype_encoder_dict[Stype.categorical.value]
    with torch.no_grad():
        numerical.lin.weight.fill_(2)
        numerical.lin.bias.fill_(3)
        numerical.na_lin.weight.fill_(4)
        numerical.na_lin.bias.fill_(-1)
        categorical.emb.weight.copy_(
            torch.tensor([[-1.], [2.], [5.], [0.]] if category_na == 'separate'
                         else [[-1.], [2.], [123.]], device=device))
    frame = TensorFrame(
        {
            Stype.numerical.value:
            torch.tensor([[1.], [float('nan')], [0.]], device=device),
            Stype.categorical.value:
            torch.tensor([[0], [-1], [99]], device=device)
        }, {
            Stype.numerical.value: ['amount'],
            Stype.categorical.value: ['category']
        })
    module = SimpleNamespace(encoder=SimpleNamespace(
        enc_dict={'transactions': encoder}))
    return module, encoder, frame


@pytest.mark.parametrize('policy_id', [
    POLICY_ID
])
def test_signed_fitted_fields_bias_missingness_and_identity(device, policy_id):
    module, encoder, frame = fixture(device)
    with FieldCapture(module) as capture:
        output = encoder(frame, torch.zeros(3, dtype=torch.long, device=device))
    assert capture.failure is None
    torch.testing.assert_close(output[:, 0],
                               torch.tensor([3., 11., 7.], device=device))
    incoming = {
        ('transactions', i): torch.tensor([6.], dtype=torch.float64,
                                          device=device)
        for i in range(3)
    }
    result = capture.allocate(incoming, h0={'transactions': output},
                              policy_id=policy_id)
    # Scalar oracle follows the actual group sum then fitted linear/embedding sum.
    eps = 1e-9
    for row, total, numeric_total, numeric_term, category_term in ((0, 3., 4.,
                                                                    2., -1.),
                                                                   (1, 11., 6.,
                                                                    4., 5.),
                                                                   (2, 7., 2.,
                                                                    0., 5.)):
        numeric_r = 6 * numeric_total / (total + eps)
        category_r = 6 * category_term / (total + eps)
        numeric_expected = numeric_r * numeric_term / (numeric_total + eps)
        category_expected = category_r * category_term / (
            category_term + (eps if category_term >= 0 else -eps))
        assert result.fields['transactions', row,
                             'amount'].item() == pytest.approx(
                                 numeric_expected, abs=1e-12)
        assert result.fields['transactions', row,
                             'category'].item() == pytest.approx(
                                 category_expected, abs=1e-12)
        allocated = sum(v for k, v in result.fields.items() if k[1] == row)
        for path, ledger in result.ledgers.items():
            if path.startswith(f'transactions/{row}/'):
                allocated += sum(b.sum() for b in ledger.biases)
                allocated += ledger.stabilization.sum() + ledger.rejected.sum(
                ) + ledger.residual.sum()
        assert allocated.item() == pytest.approx(6., abs=1e-12)
    assert result.fields['transactions', 0, 'category'] < 0
    assert result.missingness['transactions', 0, 'amount'] == 0
    assert result.missingness['transactions', 1,
                              'amount'] == result.fields['transactions', 1,
                                                         'amount']
    assert result.missingness['transactions', 2,
                              'category'] == result.fields['transactions', 2,
                                                           'category']
    assert all(result.complete.values())
    assert all(value.device == device
               for value in result.field_channels.values())
    altered = output.clone()
    altered[0] += 1
    with pytest.raises(ValueError, match='h0 does not match'):
        capture.allocate(incoming, h0={'transactions': altered},
                         policy_id=policy_id)
    invalid = {
        key: torch.ones_like(value, dtype=torch.bool)
        for key, value in incoming.items()
    }
    assert not any(
        capture.allocate(incoming, h0={
            'transactions': output
        }, invalid=invalid, policy_id=policy_id).complete.values())


@pytest.mark.parametrize('policy_id', [
    POLICY_ID
])
def test_rejection_constants_relu_and_refusals(device, policy_id):
    module, encoder, frame = fixture(device)
    numerical = encoder.stype_encoder_dict[Stype.numerical.value]
    categorical = encoder.stype_encoder_dict[Stype.categorical.value]
    # Nonzero graph input, but the numerical affine's bias cancels its value.
    with torch.no_grad():
        numerical.lin.weight.fill_(101.)
        numerical.lin.bias.fill_(-100.)
        numerical.na_lin.bias.zero_()
        categorical.emb.weight[0].fill_(2.)
    with FieldCapture(module) as capture:
        output = encoder(frame, torch.zeros(3, dtype=torch.long, device=device))
    relevance = {
        ('transactions', 0): torch.ones(1, dtype=torch.float64, device=device)
    }
    result = capture.allocate(relevance, h0={'transactions': output},
                              policy_id=policy_id)
    if policy_id == POLICY_ID:
        assert result.fields['transactions', 0, 'amount'] == 0
        assert not result.complete['transactions', 0, 'amount']
        assert result.ledgers['transactions/0/numerical'].rejection_mask.item()
    else:
        assert result.fields['transactions', 0, 'amount'] > 0
        assert all(result.complete.values())
    empty = TensorFrameEncoder('customers', {}, None, 1, device=device)
    module.encoder.enc_dict = {'customers': empty}
    with FieldCapture(module) as capture:
        output = empty(TensorFrame({}, {}, num_rows=2),
                       torch.zeros(2, dtype=torch.long, device=device))
    result = capture.allocate({('customers', 1): relevance['transactions', 0]},
                              h0={'customers': output}, policy_id=policy_id)
    assert not result.fields
    assert result.constants['customers', 1].item() == pytest.approx(
        1 / (1 + 1e-9),
        abs=1e-14, rel=1e-14)
    empty.act = torch.nn.Sigmoid()
    with pytest.raises(ValueError, match='Unsupported field adapter'):
        with FieldCapture(module):
            pass
    assert not empty._forward_hooks and not empty._forward_pre_hooks
    module, encoder, frame = fixture(device, act='relu')
    with torch.no_grad():
        encoder.stype_encoder_dict[Stype.numerical.value].lin.bias.fill_(-100)
    with FieldCapture(module) as capture:
        output = encoder(frame, torch.zeros(3, dtype=torch.long, device=device))
    with pytest.raises(ValueError, match='inactive encoder ReLU'):
        capture.allocate(relevance, h0={'transactions': output},
                         policy_id=policy_id)
    frame.col_names_dict[Stype.numerical.value] = ['wrong']
    with FieldCapture(module) as capture:
        encoder(frame, torch.zeros(3, dtype=torch.long, device=device))
    with pytest.raises(ValueError, match='order of columns'):
        capture.allocate(relevance, h0={'transactions': output},
                         policy_id=policy_id)

    # EmbeddingBag skips its padding row even when that parameter is nonzero.
    module, encoder, frame = fixture(device, category_na='zero')
    with FieldCapture(module) as capture:
        output = encoder(frame, torch.zeros(3, dtype=torch.long, device=device))
    assert capture.failure is None
    torch.testing.assert_close(output[:, 0],
                               torch.tensor([3., 6., 2.], device=device))
    result = capture.allocate(
        {('transactions', 1): relevance['transactions', 0]},
        h0={'transactions': output}, policy_id=policy_id)
    assert result.fields['transactions', 1, 'category'] == 0
    assert result.complete['transactions', 1, 'category']

    # Target exclusion metadata can name columns absent from fitted features.
    encoder.exclude_cols = ['is_fraud']
    for fused in encoder.stype_encoder_dict.values():
        fused.exclude_cols = ['is_fraud']
    with FieldCapture(module) as capture:
        repeated = encoder(frame, torch.zeros(3, dtype=torch.long,
                                              device=device))
    assert capture.failure is None
    assert torch.equal(output, repeated)
    repeated_result = capture.allocate(
        {('transactions', 1): relevance['transactions', 0]},
        h0={'transactions': repeated}, policy_id=policy_id)
    for key in result.fields:
        assert torch.equal(result.fields[key], repeated_result.fields[key])
    numerical = encoder.stype_encoder_dict[Stype.numerical.value]
    numerical.exclude_cols = ['amount']
    numerical.exclude_cols_mask.fill_(True)
    with pytest.raises(ValueError, match='Unsupported excluded field'):
        with FieldCapture(module):
            pass


@pytest.mark.parametrize('policy_id', [
    POLICY_ID, SMOOTH_FLOOR_POLICY_ID
])
@pytest.mark.parametrize('budget', [1, 1000])
@pytest.mark.parametrize('scenario',
                         ['clean', 'field_rejected', 'graph_rejected'])
@pytest.mark.parametrize('sign', [-1, 1])
def test_field_walk_composition_accounting(device, budget, scenario, sign,
                                           policy_id):
    from dataclasses import replace

    from kumo_core.explain.walk_graph import GraphCapture
    from kumo_core.explain.walk_lrp_epsilon import enumerate_canonical_walks

    from tests.test_walk_graph import b1_batch, b1_forward, b1_module

    module, encoder, frame = fixture(device)
    if scenario == 'field_rejected':
        with torch.no_grad():
            numeric = encoder.stype_encoder_dict[Stype.numerical.value]
            numeric.lin.weight.fill_(101.)
            numeric.lin.bias.fill_(-100.)
            numeric.na_lin.bias.zero_()
            encoder.stype_encoder_dict[
                Stype.categorical.value].emb.weight[0].fill_(2.)
    frame = frame[:2]
    # Reuse the fitted-encoder fixture and B1 graph, including featureless leaves.
    module.encoder.enc_dict.update({
        name:
        TensorFrameEncoder(name, {}, None, 1, device=device)
        for name in ('customer', 'article')
    })
    with FieldCapture(module) as fields:
        h0 = {
            'transactions':
            encoder(frame, torch.zeros(2, dtype=torch.long, device=device))
        }
        for name in ('customer', 'article'):
            h0[name] = module.encoder.enc_dict[name](
                TensorFrame({}, {}, num_rows=1),
                torch.zeros(1, dtype=torch.long, device=device))
    graph = b1_module(dtype=torch.float32, head_bias=True).to(device)
    batch = list(b1_batch(dtype=torch.float32))
    batch[0] = h0
    batch[1] = {key: value.to(device) for key, value in batch[1].items()}
    if scenario == 'graph_rejected':
        with torch.no_grad():
            graph.head.mlp.lins[0].bias.zero_()
            score = b1_forward(graph, batch).item()
            graph.head.mlp.lins[0].bias.fill_(-score + .01)
    with GraphCapture(graph) as capture:
        b1_forward(graph, batch)
    computation = capture.computation(preserve_device=True)
    encoded = enumerate_canonical_walks(computation, sign=sign,
                                        max_walks=budget, top_k=0,
                                        policy_id=policy_id)
    h0 = computation.h0
    result = fields.allocate_walks(encoded, h0=h0, top_k=1)
    assert result.policy_id == encoded.policy_id == policy_id
    assert result.coverage['policy_id'] == policy_id
    assert all(a.policy_id == policy_id for a in result.allocations)
    assert all(split.policy_id == policy_id
               for a in (result.merged, *result.allocations)
               for split in a.ledgers.values())
    if policy_id != POLICY_ID:
        assert result.terminals['rejected'] == 0
        assert all(result.merged.complete.values())
    assert result.encoded_walks is encoded
    assert len(result.allocations) == len(encoded.enumeration.walks)
    assert result.coverage['ranking_certified'] == (
        budget == 1000 and (scenario == 'clean' or policy_id != POLICY_ID))
    accounted = sum(result.terminals.values())
    for index, allocation in enumerate(result.allocations):
        accounted += sum(allocation.fields.values())
        assert all(value.device == device
                   for value in allocation.fields.values())
        if bool(encoded.invalid_walks[index].any()):
            assert not any(allocation.complete.values())
    accounted += sum(prefix.mass for prefix in encoded.enumeration.prefixes)
    torch.testing.assert_close(accounted, encoded.merged.score, rtol=1e-12,
                               atol=1e-13)
    for side, orientation in (('positive', 1), ('negative', -1)):
        coverage = result.coverage[side]
        found = sum(
            max(orientation * float(value), 0.)
            for allocation in result.allocations
            for value in allocation.fields.values())
        assert coverage['enumerated'] == pytest.approx(found, abs=1e-13)
        if budget == 1:
            assert coverage['total'] is None
            assert coverage['fraction'] is None
            assert coverage['omitted'] is None
        else:
            assert coverage['total'] == pytest.approx(found, abs=1e-13)
    if budget == 1000:
        for key, expected in result.merged.fields.items():
            actual = sum(
                allocation.fields.get(key, expected.new_zeros(()))
                for allocation in result.allocations)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
        expected_terminals = {
            key: value.clone()
            for key, value in encoded.terminals.items()
        }
        for split in result.merged.ledgers.values():
            expected_terminals['bias'] += sum(value.sum()
                                              for value in split.biases)
            expected_terminals['constant'] += sum(value.sum()
                                                  for value in split.constants)
            for key in ('stabilization', 'rejected', 'residual'):
                expected_terminals[key] += getattr(split, key).sum()
        for key in expected_terminals:
            torch.testing.assert_close(result.terminals[key],
                                       expected_terminals[key], rtol=1e-12,
                                       atol=1e-13)
        assert result.terminals['constant'] != 0 or scenario == 'graph_rejected'
    changed = {name: value.clone() for name, value in h0.items()}
    changed['transactions'][0] += 1
    with pytest.raises(ValueError, match='h0 does not match'):
        fields.allocate_walks(encoded, h0=changed)
    with pytest.raises(ValueError, match='misaligned'):
        fields.allocate_walks(replace(encoded, invalid_walks=(None, )), h0=h0)
    for top_k in (-1, True):
        with pytest.raises(ValueError, match='top_k'):
            fields.allocate_walks(encoded, h0=h0, top_k=top_k)

    with pytest.raises(ValueError, match='Unknown LRP policy'):
        fields.allocate({}, h0=h0, policy_id='unregistered')
    with pytest.raises(ValueError, match='Unknown LRP policy'):
        enumerate_canonical_walks(computation, policy_id='unregistered')
    with pytest.raises(ValueError, match='Unknown LRP policy'):
        fields.allocate_walks(
            replace(encoded, merged=replace(encoded.merged,
                                            policy_id='unregistered')), h0=h0)

    # A cloned input is numerically compatible, not proof of capture identity.
    clones = {name: value.clone() for name, value in h0.items()}
    fields.allocate_walks(encoded, h0=clones, top_k=0)
    original = h0['transactions'].clone()
    h0['transactions'].add_(1)
    assert torch.equal(encoded.captured_h0['transactions'], original)
    with pytest.raises(ValueError, match='h0 does not match canonical'):
        fields.allocate_walks(encoded, h0=h0)
    h0['transactions'].copy_(original)
    # Even paired caller/encoder mutations cannot replace the bound graph input.
    saved_output = fields.tables['transactions']['output']
    fields.tables['transactions']['output'] = changed['transactions'].clone()
    with pytest.raises(ValueError, match='h0 does not match canonical'):
        fields.allocate_walks(encoded, h0=changed)
    fields.tables['transactions']['output'] = saved_output
    for mutation in ('missing', 'extra', 'dtype', 'shape', 'nonfinite'):
        altered = {name: value.clone() for name, value in h0.items()}
        if mutation == 'missing':
            del altered['article']
        elif mutation == 'extra':
            altered['unexpected'] = original
        elif mutation == 'dtype':
            altered['transactions'] = original.float()
        elif mutation == 'shape':
            altered['transactions'] = original[:1]
        else:
            altered['transactions'][0] = float('inf')
        with pytest.raises(ValueError, match='h0.*(does not match|table set)'):
            fields.allocate_walks(encoded, h0=altered, top_k=0)
    table = fields.tables.pop('article')
    with pytest.raises(ValueError, match='Fitted capture table set'):
        fields.allocate_walks(encoded, h0=h0)
    fields.tables['article'] = table
    fields.tables['unexpected'] = table
    with pytest.raises(ValueError, match='Fitted capture table set'):
        fields.allocate_walks(encoded, h0=h0)
    del fields.tables['unexpected']
    # NodeGNNModule separately encodes its task table outside graph h0.
    fields.tables['task'] = table
    bound_task = replace(encoded, train_table_name='task')
    with_task = fields.allocate_walks(bound_task, h0=h0)
    assert all(key[0] != 'task' for key in with_task.merged.fields)
    assert all(key[0] != 'task' for key in with_task.merged.constants)
    del fields.tables['task']
