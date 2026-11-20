# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent rational fixtures for the unregistered epsilon primitive."""
from fractions import Fraction

import pytest
import torch
from kumo_core.explain.walk_lrp_epsilon import (
    POLICY_ID,
    SMOOTH_FLOOR_POLICY_ID,
    canonical_epsilon_split,
)


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; GPU primitive remains unqualified')
    return torch.device('cuda:0' if request.param == 'cuda' else 'cpu')


def test_rational_split_and_device(device):
    # Exact scalar oracle: S=2-1+3-1=3, R=-7, epsilon=1/10**9.
    z = torch.tensor([[2.], [-1.]], dtype=torch.float64, device=device)
    b = torch.tensor([3.], dtype=torch.float64, device=device)
    c = torch.tensor([-1.], dtype=torch.float64, device=device)
    r = torch.tensor([-7.], dtype=torch.float64, device=device)
    result = canonical_epsilon_split([z], [b], r, constants=[c])
    denominator = Fraction(3) + Fraction(1, 10**9)
    expected = [
        float(Fraction(-7) * term / denominator)
        for term in (2, -1, 3, -1, Fraction(1, 10**9))
    ]
    actual = torch.cat((result.contributions[0].flatten(), result.biases[0],
                        result.constants[0], result.stabilization))
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=torch.float64, device=device),
        rtol=1e-14, atol=1e-14)
    assert not result.rejection_mask.any()
    assert result.cancellation.item() == pytest.approx(7 / 3)
    assert result.rejected.item() == 0
    assert abs(result.residual.item()) < 1e-14
    for value in (*result.contributions, *result.biases, *result.constants,
                  result.stabilization, result.rejected, result.residual,
                  result.rejection_mask, result.cancellation):
        assert value.device == device and (value.dtype == torch.float64
                                           or value.dtype == torch.bool)


def test_threshold_zero_and_unrelated_columns(device):
    # Columns: K=100, K=101, canceled zero, all-zero, negative S.
    z = torch.tensor([[50.5, 51., 1., -0., -2.], [-49.5, -50., -1., 0., -1.]],
                     dtype=torch.float64, device=device)
    r = torch.tensor([2., 3., 0., -4., 5.], dtype=torch.float64, device=device)
    result = canonical_epsilon_split([z], [], r)
    assert result.rejection_mask.tolist() == [False, True, True, False, False]
    assert result.cancellation.tolist() == [100., 101., float('inf'), 0., 1.]
    assert result.rejected.tolist() == [0., 3., 0., 0., 0.]
    assert torch.equal(result.contributions[0][:, 1:4],
                       torch.zeros_like(z[:, 1:4]))
    assert result.stabilization[3].item() == -4.
    assert result.stabilization[4].item() == pytest.approx(5e-9 / (3 + 1e-9))
    assert result.contributions[0][:, 4].sum().item() > 4.99
    torch.testing.assert_close(
        result.contributions[0].sum(0) + result.rejected +
        result.stabilization + result.residual, r)


def test_bias_constants_empty_and_boundary(device):
    r = torch.tensor([1.], dtype=torch.float64, device=device)
    one = torch.ones((1, 1), dtype=torch.float64, device=device)
    for kwargs in ({'biases': [-r]}, {'biases': [], 'constants': [-r]}):
        result = canonical_epsilon_split([one], relevance=r, **kwargs)
        assert result.rejection_mask.item()
        assert result.rejected.item() == 1.
        assert result.stabilization.item() == 0.
    empty = canonical_epsilon_split([], [], r)
    assert empty.stabilization.item() == 1. and empty.residual.item() == 0.
    # Keep S=1 exactly while incrementing both opposing magnitudes.
    a = torch.nextafter(
        torch.tensor(50.5, dtype=torch.float64, device=device),
        torch.tensor(float('inf'), dtype=torch.float64, device=device))
    above = canonical_epsilon_split([torch.stack((a, 1 - a)).reshape(2, 1)], [],
                                    r)
    assert above.rejection_mask.item() and above.cancellation.item() > 100.


@pytest.mark.parametrize('policy_id', [
    POLICY_ID, SMOOTH_FLOOR_POLICY_ID
])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_input_refused(device, bad, policy_id):
    r = torch.zeros(1, dtype=torch.float64, device=device)
    invalid = torch.full_like(r, bad)
    for z, b, c, relevance in (([invalid.reshape(1, 1)], [], [],
                                r), ([], [invalid], [], r),
                               ([], [], [invalid], r), ([], [], [], invalid)):
        with pytest.raises(ValueError, match='nonfinite LRP input'):
            canonical_epsilon_split(z, b, relevance, constants=c,
                                    policy_id=policy_id)


@pytest.mark.parametrize('policy_id', [
    POLICY_ID, SMOOTH_FLOOR_POLICY_ID
])
def test_malformed_and_overflow_refused(device, policy_id):
    r = torch.ones(1, dtype=torch.float64, device=device)
    for value in (torch.ones(1, dtype=torch.float32, device=device),
                  torch.ones(2, dtype=torch.float64,
                             device=device), r.reshape(1, 1)):
        with pytest.raises(ValueError, match='matching shape'):
            canonical_epsilon_split([], [value], r, policy_id=policy_id)
    for values in ([1e308, 1e308], [1e308, -1e308]):
        with pytest.raises(ValueError,
                           match='nonfinite LRP total or absolute mass'):
            canonical_epsilon_split([
                torch.tensor(values, dtype=torch.float64,
                             device=device).reshape(2, 1)
            ], [], r, policy_id=policy_id)
    opposing = [
        torch.tensor([[50.5], [-49.5]], dtype=torch.float64, device=device)
    ]
    with pytest.raises(ValueError, match='nonfinite LRP allocation'):
        canonical_epsilon_split(opposing, [], r * 1e308,
                                policy_id=policy_id)
    if device.type == 'cuda':
        with pytest.raises(ValueError, match='matching shape'):
            canonical_epsilon_split([], [r.cpu()], r, policy_id=policy_id)


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('rejected', [False, True])
def test_captured_graph_propagation_device_and_rejection(
        device, sign, rejected):
    from kumo_core.explain.walk_graph import GraphCapture
    from kumo_core.explain.walk_lrp_epsilon import propagate

    from tests.test_walk_graph import b1_batch, b1_forward, b1_module

    module = b1_module(head_bias=rejected).to(device)
    batch = list(b1_batch())
    batch[:2] = [{
        key: tensor.to(device)
        for key, tensor in part.items()
    } for part in batch[:2]]
    if rejected:
        with torch.no_grad():
            module.head.mlp.lins[0].bias.fill_(-8.99)
    expected = b1_forward(module, batch)
    with GraphCapture(module) as capture:
        actual = b1_forward(module, batch)
    assert torch.equal(actual, expected)
    computation = capture.computation(preserve_device=True)
    result = propagate(computation, sign=sign)
    assert result.score.item() == sign * actual.item()
    assert set(result.h0) == {('customer', 0), ('transactions', 0),
                              ('transactions', 1), ('article', 0)}
    assert all(value.device == device for value in result.h0.values())
    torch.testing.assert_close(sum(result.ledger().values()), result.score,
                               rtol=1e-13, atol=1e-13)
    assert all(
        bool(mask.all()) == rejected for mask in result.invalid_h0.values())
    head = result.operations[0]
    assert head.split.rejection_mask.item() == rejected
    if rejected:
        assert result.ledger()['encoded_inputs'].item() == 0
        assert result.ledger()['rejected'].item() == result.score.item()
    else:
        # Two MP layers, all-positive weights, with a mean relation and
        # distinct transaction occurrences. Analytic leaf allocation tends
        # to [2, 2, 3, 2] in customer, t0, t1, article order as epsilon -> 0.
        expected_leaf = {
            ('customer', 0): 2.,
            ('transactions', 0): 2.,
            ('transactions', 1): 3.,
            ('article', 0): 2.
        }
        for node, value in result.h0.items():
            assert value.item() == pytest.approx(sign * expected_leaf[node],
                                                 abs=1e-8)
    for operation in result.operations:
        assert operation.relevance.device == device
        assert all(z.device == device for z in operation.contributions)
        for contribution in operation.split.contributions:
            assert contribution.device == device
            assert torch.equal(
                contribution[:, operation.split.rejection_mask],
                torch.zeros_like(contribution[:,
                                              operation.split.rejection_mask]))
        if operation.kind == 'message_passing':
            assert len(operation.routes) == len(operation.contributions)
            assert all(route.layer == operation.index
                       and route.destination == operation.node
                       for route in operation.routes)
    assert torch.equal(computation.forward_piece().z_piece, actual.flatten())
    if device.type == 'cuda':
        with pytest.raises(ValueError, match='preserve the capture device'):
            propagate(capture.computation())


@pytest.mark.parametrize('policy_id', [
    POLICY_ID, SMOOTH_FLOOR_POLICY_ID
])
@pytest.mark.parametrize('act', ['relu', None])
def test_canonical_preprocessing_and_signed_biases(act, policy_id):
    from kumo_core.explain.walk_graph import GraphCapture
    from kumo_core.explain.walk_lrp_epsilon import propagate

    from tests.test_walk_graph import preprocessing_module

    module, run = preprocessing_module(act=act)
    with GraphCapture(module) as capture:
        run()
    result = propagate(capture.computation(preserve_device=True),
                       policy_id=policy_id)
    assert {op.kind
            for op in result.operations
            } == {'head', 'message_passing', 'preprocessing'}
    assert len([op for op in result.operations if op.kind == 'head']) == 2
    assert any(
        bool((value < 0).any()) for op in result.operations
        for value in op.split.biases)
    torch.testing.assert_close(sum(result.ledger().values()), result.score,
                               rtol=1e-13, atol=1e-13)


def test_canonical_refuses_unqualified_capture_and_bad_selection():
    from kumo_core.explain.walk_graph import GraphCapture
    from kumo_core.explain.walk_lrp_epsilon import propagate

    from tests.test_walk_graph import b1_batch, b1_forward, b1_module

    module, batch = b1_module(), b1_batch()
    with GraphCapture(module, qualify=False) as capture:
        b1_forward(module, batch)
    computation = capture.computation(preserve_device=True)
    with pytest.raises(ValueError, match='structural qualification'):
        propagate(computation)
    for kwargs in ({
            'sign': True
    }, {
            'sign': 0
    }, {
            'output_index': True
    }, {
            'output_index': 1
    }):
        with pytest.raises(ValueError):
            propagate(computation, **kwargs)


def test_preserved_capture_refuses_mixed_devices(device):
    if device.type != 'cuda':
        pytest.skip('Mixed-device validation requires CUDA')
    from kumo_core.explain.walk_graph import GraphCapture

    from tests.test_walk_graph import b1_batch, b1_forward, b1_module

    module = b1_module().to(device)
    batch = list(b1_batch())
    batch[:2] = [{
        key: tensor.to(device)
        for key, tensor in part.items()
    } for part in batch[:2]]
    with GraphCapture(module) as capture:
        b1_forward(module, batch)
    original = capture.h0['customer']
    capture.h0['customer'] = original.cpu()
    with pytest.raises(ValueError, match='multiple devices'):
        capture.computation(preserve_device=True)
    capture.h0['customer'] = original
    capture.layers[0].relations[0].edge_index = (
        capture.layers[0].relations[0].edge_index.cpu())
    with pytest.raises(ValueError, match='edges differ'):
        capture.computation(preserve_device=True)


@pytest.mark.parametrize('budget', [1, 4, 1000])
@pytest.mark.parametrize('rejected', [False, True])
def test_canonical_walk_budget_accounting_and_ancestry(device, budget,
                                                       rejected):
    from kumo_core.explain.walk_graph import GraphCapture
    from kumo_core.explain.walk_lrp_epsilon import enumerate_canonical_walks

    from tests.test_walk_graph import b1_batch, b1_forward, b1_module

    module = b1_module(head_bias=rejected).to(device)
    batch = list(b1_batch())
    batch[:2] = [{
        k: v.to(device)
        for k, v in part.items()
    } for part in batch[:2]]
    if rejected:
        with torch.no_grad():
            module.head.mlp.lins[0].bias.fill_(-8.99)
    with GraphCapture(module) as capture:
        b1_forward(module, batch)
    result = enumerate_canonical_walks(
        capture.computation(preserve_device=True), max_walks=budget, top_k=1)
    enumeration = result.enumeration
    assert enumeration.expanded_routes <= budget
    assert enumeration.complete == (budget == 1000)
    assert len(result.invalid_walks) == len(enumeration.walks)
    assert len(result.invalid_prefixes) == len(enumeration.prefixes)
    for mask in (*result.invalid_walks, *result.invalid_prefixes):
        assert mask.device == device and bool(mask.all()) == rejected
    accounted = (sum(result.terminals.values()) +
                 sum(w.relevance for w in enumeration.walks) +
                 sum(p.mass for p in enumeration.prefixes))
    torch.testing.assert_close(accounted, result.merged.score, rtol=1e-13,
                               atol=1e-13)
    for side in ('positive', 'negative'):
        coverage = result.coverage[side]
        if not enumeration.complete:
            assert coverage['total'] is None
            assert coverage['fraction'] is None
            assert coverage['omitted'] is None
    assert result.coverage['ranking_certified'] == (enumeration.complete
                                                    and not rejected)
    assert enumeration.intermediate_state_abs_mass is not None
    assert enumeration.intermediate_state_abs_mass >= abs(result.merged.score)
    if enumeration.complete:
        by_source = {
            key: torch.zeros_like(value)
            for key, value in result.merged.h0.items()
        }
        for walk in enumeration.walks:
            by_source[walk.source] += walk.input_relevance
            assert walk.input_relevance.device == device
            assert [r.layer for r in walk.steps] == [1, 2]
            assert walk.steps[0].source == walk.source
            assert walk.steps[0].destination == walk.steps[1].source
        for key in by_source:
            torch.testing.assert_close(by_source[key], result.merged.h0[key],
                                       rtol=1e-13, atol=1e-13)
        for key in result.terminals:
            torch.testing.assert_close(result.terminals[key],
                                       result.merged.ledger()[key], rtol=1e-13,
                                       atol=1e-13)
        if rejected:
            assert all(w.relevance == 0 for w in enumeration.walks)


def test_canonical_walk_signed_display_and_preprocessing():
    from kumo_core.explain.walk_graph import GraphCapture
    from kumo_core.explain.walk_lrp_epsilon import enumerate_canonical_walks

    from tests.test_walk_graph import preprocessing_module

    module, run = preprocessing_module(act=None)
    with GraphCapture(module) as capture:
        run()
    computation = capture.computation(preserve_device=True)
    result = enumerate_canonical_walks(computation, max_walks=1000, top_k=0)
    assert result.enumeration.complete and result.displayed_indices == ()
    assert all(result.coverage[side]['displayed'] == 0
               for side in ('positive', 'negative'))
    assert sum(w.relevance for w in result.enumeration.walks) == pytest.approx(
        result.merged.ledger()['encoded_inputs'].item(), abs=1e-12)
    inverted = enumerate_canonical_walks(computation, sign=-1, max_walks=1000,
                                         top_k=1000)
    for original, opposite in zip(result.enumeration.walks,
                                  inverted.enumeration.walks):
        assert original.relevance == pytest.approx(-opposite.relevance,
                                                   abs=1e-12)
    for side, opposite in (('positive', 'negative'), ('negative', 'positive')):
        assert result.coverage[side]['total'] == pytest.approx(
            inverted.coverage[opposite]['total'], abs=1e-12)
    for kwargs in ({
            'max_walks': True
    }, {
            'max_walks': 0
    }, {
            'top_k': -1
    }, {
            'top_k': True
    }):
        with pytest.raises(ValueError):
            enumerate_canonical_walks(computation, **kwargs)


@pytest.mark.parametrize('magnitude', [1e-300, 1e-9, 1., 1e300])
@pytest.mark.parametrize('orientation', [-1, 1])
def test_smooth_floor_extremes_and_retention(device, magnitude, orientation):
    # Reuse the rational split oracle at scales where direct squares fail.
    z = torch.tensor([[magnitude], [magnitude * 2]], dtype=torch.float64,
                     device=device) * orientation
    r = torch.tensor([-3.], dtype=torch.float64, device=device)
    result = canonical_epsilon_split([z], [], r,
                                     policy_id=SMOOTH_FLOOR_POLICY_ID)
    terms = [Fraction(float(v)) for v in z[:, 0]]
    total, mass = sum(terms), sum(abs(v) for v in terms)
    q = mass / 100 + Fraction(1, 10**9)
    coefficients = [v * total / (total**2 + q**2) for v in terms]
    torch.testing.assert_close(
        result.contributions[0][:, 0],
        r.new_tensor([float(-3 * v) for v in coefficients]), rtol=1e-14,
        atol=0.)
    assert result.stabilization.item() == pytest.approx(
        float(-3 * q**2 / (total**2 + q**2)), rel=1e-14, abs=0.)
    if magnitude >= 1:
        # At K=1 the retained fraction approaches 1/(1+1/10000).
        assert result.contributions[0].sum() / r[0] > .9998


@pytest.mark.parametrize('delta', [-1e-12, 0., 1e-12])
def test_smooth_cancellation_accounting_and_continuity(device, delta):
    terms = torch.tensor([[1.], [-1. + delta]], dtype=torch.float64, device=device)
    relevance = torch.tensor([3.], dtype=torch.float64, device=device)
    result = canonical_epsilon_split([terms], [], relevance,
                                     policy_id=SMOOTH_FLOOR_POLICY_ID)
    exact = [Fraction(float(v)) for v in terms[:, 0]]
    total, mass = sum(exact), sum(abs(v) for v in exact)
    q = mass / 100 + Fraction(1, 10**9)
    expected = [float(3 * v * total / (total**2 + q**2)) for v in exact]
    torch.testing.assert_close(result.contributions[0][:, 0],
                               relevance.new_tensor(expected), atol=1e-20, rtol=1e-14)
    assert result.contributions[0].abs().max() < 1e-7
    assert (result.contributions[0].sum() + result.stabilization.sum()
            + result.residual.sum()).item() == pytest.approx(3., abs=1e-14)
    assert not result.rejection_mask.any()
