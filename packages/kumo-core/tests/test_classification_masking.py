# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cumulative policy and fixed-stand-in companion contracts."""
import copy

import pytest
import torch
from kumo_core.explain.classification_masking import (
    FeasibilityCompanion,
    fidelity,
    fit_feasibility,
    verified_greedy,
)

KEY = (1, ('events', 'owner_id', 'owners'))
ORIGINAL = {'raw': 1., 'served': .7}


def test_cumulative_rollback_uses_original_and_has_twelve_trial_budget():
    ranked = [(i / 20, KEY, i) for i in range(15)]
    seen = []

    def evaluate(proposed):
        seen.append(tuple(edge for _, edge in proposed))
        return {'raw': 1.03 if len(proposed) == 1 else 1.06, 'served': .7}

    result = verified_greedy(ranked, ORIGINAL, ORIGINAL, evaluate, .5)
    assert len(seen) == 12
    assert seen[0] == (0,) and seen[1] == (0, 1) and seen[-1] == (0, 11)
    assert result['accepted_ranks'] == [1]
    assert result['rejected_attempts'] == 11
    assert result['selected']['raw'] == 1.03
    assert all(t['accepted_ranks_after'] == [1] for t in result['attempts'])


@pytest.mark.parametrize('n', [0, 1, 3, 15])
def test_ties_insufficient_candidates_and_verified_keepall(n):
    ranked = [(.2, KEY, i) for i in reversed(range(n))]
    result = verified_greedy(ranked, ORIGINAL, ORIGINAL,
                             lambda _: {'raw': 2., 'served': .7}, .5)
    assert result['fallback'] and result['selected'] == ORIGINAL
    assert [t['edge_column'] for t in result['attempts']] == list(range(min(n, 12)))
    assert result['selected_mask'] == []


@pytest.mark.parametrize('result,threshold,passed', [
    ({'raw': .05, 'served': .01}, 0., True),
    ({'raw': .0500000001, 'served': .01}, 0., False),
    ({'raw': 0., 'served': .0100000001}, 0., False),
    ({'raw': 0., 'served': .01}, .01, False),
])
def test_fidelity_boundaries_and_explicit_decision_threshold(result, threshold, passed):
    assert fidelity({'raw': 0., 'served': 0.}, result, threshold)['fidelity_pass'] is passed


@pytest.mark.parametrize('fault', ['duplicate', 'nan', 'edge', 'threshold', 'keepall', 'trial'])
def test_invalid_inputs_withhold_output(fault):
    ranked = [(.2, KEY, 0)]
    threshold = .5
    keepall = ORIGINAL
    evaluate = lambda _: ORIGINAL
    if fault == 'duplicate':
        ranked *= 2
    elif fault == 'nan':
        ranked = [(float('nan'), KEY, 0)]
    elif fault == 'edge':
        ranked = [(.2, KEY, -1)]
    elif fault == 'threshold':
        threshold = float('nan')
    elif fault == 'keepall':
        keepall = {'raw': 1.1, 'served': .7}
    else:
        evaluate = lambda _: {'raw': float('nan'), 'served': .7}
    with pytest.raises(ValueError):
        verified_greedy(ranked, ORIGINAL, keepall, evaluate, threshold)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_case_balanced_fit_only_changes_owned_gate_networks(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('Fresh GPU qualification pending')
    standins = {1: torch.tensor([2., -1.], device=device)}
    torch.manual_seed(36)
    companion = FeasibilityCompanion([KEY], 2, standins).to(device)
    source = standins[1].clone()
    state = copy.deepcopy(companion.state_dict())
    contexts = torch.tensor([[1., 2., 3., 4., 5., 6.]], device=device)
    cases = [({KEY: contexts}, {KEY: torch.tensor([0.], device=device)}),
             ({KEY: contexts.repeat(3, 1)}, {KEY: torch.ones(3, device=device)})]
    with torch.no_grad():
        gate = companion(cases[0][0])[0][KEY]
        expected = (torch.nn.functional.binary_cross_entropy(gate, gate.new_zeros(1))
                    + torch.nn.functional.binary_cross_entropy(gate, gate.new_ones(1))) / 2
    optimizer, curve = fit_feasibility(companion, cases, epochs=2)
    assert curve[0]['loss'] == pytest.approx(float(expected))
    assert torch.equal(companion.standins['1'], state['standins.1'])
    assert torch.equal(standins[1], source)
    assert companion.standins['1'].grad is None
    assert any(not torch.equal(value, state[name]) for name, value in companion.state_dict().items()
               if name.startswith('networks.'))
    assert len(optimizer['param_groups'][0]['params']) == len(list(companion.networks.parameters()))
    assert not companion.training
    standins[1].add_(1.)
    assert torch.equal(companion.standins['1'], source)
    with pytest.raises(ValueError, match='Invalid detached'):
        companion({KEY: contexts.fill_(float('nan'))})
